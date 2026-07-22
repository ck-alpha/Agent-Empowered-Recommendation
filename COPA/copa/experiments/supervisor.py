"""Resumable tmux-friendly supervisor for the complete COPA core study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervise the COPA core experiment")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase1-config", default="COPA/configs/core_experiment.yaml")
    parser.add_argument("--phase2-config", default="COPA/configs/phase2_qwen.yaml")
    parser.add_argument("--phase3-config", default="COPA/configs/phase3_agent.yaml")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cohort-seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResourceMonitor:
    def __init__(self, output: Path, stage_getter, interval: float = 10.0) -> None:
        self.output = output
        self.stage_getter = stage_getter
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 2)

    def _process_usage(self) -> tuple[float, float]:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,rss=,pcpu="],
            capture_output=True,
            text=True,
            check=False,
        )
        rows = []
        for line in result.stdout.splitlines():
            try:
                pid, ppid, rss, cpu = line.split()
                rows.append((int(pid), int(ppid), float(rss), float(cpu)))
            except ValueError:
                continue
        descendants = {os.getpid()}
        changed = True
        while changed:
            changed = False
            for pid, ppid, _, _ in rows:
                if ppid in descendants and pid not in descendants:
                    descendants.add(pid)
                    changed = True
        rss_mb = sum(row[2] for row in rows if row[0] in descendants) / 1024
        cpu = sum(row[3] for row in rows if row[0] in descendants)
        return cpu, rss_mb

    @staticmethod
    def _system_memory() -> tuple[float, float]:
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            values[key] = float(value.strip().split()[0]) / 1024
        return values.get("MemTotal", 0.0), values.get("MemAvailable", 0.0)

    @staticmethod
    def _gpu() -> tuple[float | None, float | None, float | None, float | None]:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return None, None, None, None
        try:
            values = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
            return tuple(values)  # type: ignore[return-value]
        except ValueError:
            return None, None, None, None

    def _run(self) -> None:
        fields = [
            "timestamp",
            "stage",
            "cpu_percent",
            "rss_mb",
            "system_memory_total_mb",
            "system_memory_available_mb",
            "disk_free_gb",
            "gpu_util_percent",
            "gpu_memory_mb",
            "gpu_temperature_c",
            "gpu_power_w",
        ]
        exists = self.output.exists() and self.output.stat().st_size > 0
        with self.output.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            while not self.stop_event.is_set():
                cpu, rss = self._process_usage()
                memory_total, memory_available = self._system_memory()
                disk = shutil.disk_usage(self.output.parent)
                gpu_util, gpu_memory, gpu_temperature, gpu_power = self._gpu()
                writer.writerow(
                    {
                        "timestamp": utc_now(),
                        "stage": self.stage_getter(),
                        "cpu_percent": cpu,
                        "rss_mb": rss,
                        "system_memory_total_mb": memory_total,
                        "system_memory_available_mb": memory_available,
                        "disk_free_gb": disk.free / (1024**3),
                        "gpu_util_percent": gpu_util,
                        "gpu_memory_mb": gpu_memory,
                        "gpu_temperature_c": gpu_temperature,
                        "gpu_power_w": gpu_power,
                    }
                )
                handle.flush()
                self.stop_event.wait(self.interval)


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(args.output_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(exist_ok=True)
        (self.root / "stages").mkdir(exist_ok=True)
        (self.root / "configs").mkdir(exist_ok=True)
        self.stage = "initializing"
        self.monitor = ResourceMonitor(self.root / "resource_usage.csv", lambda: self.stage)
        self.commands: list[str] = []

    def record_status(self, payload: dict[str, Any]) -> None:
        with (self.root / "stage_status.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": utc_now(), **payload}, ensure_ascii=False) + "\n")

    def marker(self, stage: str) -> Path:
        return self.root / "stages" / f"{stage}.complete.json"

    def run_command(self, stage: str, command: Sequence[str]) -> None:
        rendered = " ".join(map(str, command))
        self.commands.append(rendered)
        log_path = self.root / "logs" / f"{stage}.log"
        self.record_status({"stage": stage, "status": "started", "command": rendered})
        started = time.perf_counter()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] COMMAND {rendered}\n")
            log.flush()
            process = subprocess.Popen(
                list(command),
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            code = process.wait()
        duration = time.perf_counter() - started
        self.record_status(
            {"stage": stage, "status": "success" if code == 0 else "failed", "exit_code": code, "duration_seconds": duration}
        )
        if code != 0:
            raise RuntimeError(f"stage {stage} failed with exit code {code}; see {log_path}")

    def complete(self, stage: str, payload: dict[str, Any] | None = None) -> None:
        marker = self.marker(stage)
        marker.write_text(
            json.dumps({"stage": stage, "completed_at": utc_now(), **(payload or {})}, indent=2) + "\n",
            encoding="utf-8",
        )

    def should_skip(self, stage: str) -> bool:
        return self.args.resume and self.marker(stage).exists()

    def preflight(self) -> None:
        stage = "preflight"
        self.stage = stage
        if self.should_skip(stage):
            return
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], text=True)
        git_diff = subprocess.check_output(["git", "diff", "--binary"])
        nvidia = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        if nvidia.returncode != 0:
            raise RuntimeError(f"GPU preflight failed: {nvidia.stderr.strip()}")
        tags = subprocess.run(
            ["curl", "-s", "http://127.0.0.1:11434/api/tags"],
            capture_output=True,
            text=True,
            check=False,
        )
        if tags.returncode != 0 or "qwen2.5:14b" not in tags.stdout:
            raise RuntimeError("Ollama qwen2.5:14b is not available at 127.0.0.1:11434")
        source_hashes = {}
        tracked_inputs = [
            *Path("COPA").glob("copa/**/*.py"),
            *Path("COPA").glob("configs/*.yaml"),
            *Path("COPA").glob("evaluation/*.jsonl"),
            *Path("COPA").glob("docs/*.md"),
        ]
        for path in sorted(set(tracked_inputs)):
            source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        dependency_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        manifest = {
            "created_at": utc_now(),
            "git_commit": git_commit,
            "git_status": git_status.splitlines(),
            "git_diff_sha256": hashlib.sha256(git_diff).hexdigest(),
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "nvidia_smi": nvidia.stdout,
            "ollama_tags": json.loads(tags.stdout),
            "source_hashes": source_hashes,
            "prompt_versions": {"phase2": "constraint_compiler_v4", "phase3": "constraint_compiler_v4", "planner": "planner_v1", "repair": "repair_v1"},
        }
        (self.root / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (self.root / "workspace.patch").write_bytes(git_diff)
        (self.root / "environment_freeze.txt").write_text(
            dependency_freeze, encoding="utf-8"
        )
        for label, source in {
            "phase1_input.yaml": self.args.phase1_config,
            "phase2_input.yaml": self.args.phase2_config,
            "phase3_input.yaml": self.args.phase3_config,
        }.items():
            shutil.copy2(source, self.root / "configs" / label)
        self.complete(stage)

    def tests(self) -> None:
        stage = "tests"
        self.stage = stage
        if self.should_skip(stage):
            return
        self.run_command(stage, [sys.executable, "-m", "pytest", "COPA/tests", "-q"])
        self.complete(stage)

    def smoke(self) -> None:
        stage = "smoke_phase1"
        self.stage = stage
        if not self.args.skip_smoke and not self.should_skip(stage):
            synthetic_dir = self.root / "smoke" / "synthetic"
            self.run_command(
                stage,
                [sys.executable, "-m", "copa.experiments.run_phase1", "--config", "COPA/configs/smoke.yaml", "--experiment", "all", "--output-dir", str(synthetic_dir)],
            )
            config = yaml.safe_load(Path(self.args.phase1_config).read_text(encoding="utf-8"))
            config["dataset"]["num_users"] = 1
            config["dataset"]["candidate_k"] = 30
            config["seeds"] = [42]
            config["workers"] = 1
            config["optimization"].update({"top_k": 5, "population_size": 8, "generations": 1})
            beauty_config = self.root / "configs" / "beauty_smoke.yaml"
            beauty_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            self.run_command(
                stage,
                [sys.executable, "-m", "copa.experiments.run_core", "--config", str(beauty_config), "--output-dir", str(self.root / "smoke" / "beauty"), "--workers", "1", "--cohort-seed", "42", "--resume"],
            )
            self.complete(stage)

        qwen_stage = "smoke_qwen8"
        self.stage = qwen_stage
        if not self.args.skip_smoke and not self.should_skip(qwen_stage):
            self.run_command(
                qwen_stage,
                [sys.executable, "-m", "copa.phase2.cli", "--config", self.args.phase2_config, "evaluate", "--gold", "COPA/evaluation/constraint_compiler_gold.jsonl", "--case-ids", "zh01,zh11,zh15,amb01,en01,en11,en15,amb06", "--output-dir", str(self.root / "smoke" / "qwen8")],
            )
            if len(pd.read_csv(self.root / "smoke" / "qwen8" / "compiler_case_results.csv")) != 8:
                raise RuntimeError("Qwen smoke did not produce 8 cases")
            self.complete(qwen_stage)

    def phase1(self) -> None:
        stage = "phase1"
        self.stage = stage
        if self.should_skip(stage):
            return
        output = self.root / "phase1"
        self.run_command(
            stage,
            [sys.executable, "-m", "copa.experiments.run_core", "--config", self.args.phase1_config, "--suite", "core", "--output-dir", str(output), "--workers", str(self.args.workers), "--cohort-seed", str(self.args.cohort_seed), "--resume"],
        )
        rows = len(pd.read_csv(output / "per_user_metrics.csv"))
        if rows != 2400:
            raise RuntimeError(f"Phase 1 acceptance expected 2400 rows, got {rows}")
        self.complete(stage, {"record_count": rows})

    def _attempt_dir(self, phase: str, repeat: int) -> Path:
        repeat_root = self.root / phase / f"repeat_{repeat}"
        repeat_root.mkdir(parents=True, exist_ok=True)
        existing = sorted(repeat_root.glob("attempt_*"))
        attempt = len(existing) + 1
        path = repeat_root / f"attempt_{attempt}"
        path.mkdir(parents=True)
        return path

    @staticmethod
    def _publish(attempt: Path, repeat_root: Path, filenames: Sequence[str]) -> None:
        for filename in filenames:
            source = attempt / filename
            if not source.exists():
                raise FileNotFoundError(source)
            destination = repeat_root / filename
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copy2(source, destination)
        (repeat_root / "selected_attempt.json").write_text(
            json.dumps({"attempt": attempt.name, "selected_at": utc_now()}, indent=2) + "\n",
            encoding="utf-8",
        )

    def phase2(self) -> None:
        for repeat in range(1, self.args.repeats + 1):
            stage = f"phase2_repeat_{repeat}"
            self.stage = stage
            if self.should_skip(stage):
                continue
            attempt = self._attempt_dir("phase2", repeat)
            self.run_command(
                stage,
                [sys.executable, "-m", "copa.phase2.cli", "--config", self.args.phase2_config, "evaluate", "--gold", "COPA/evaluation/constraint_compiler_gold.jsonl", "--output-dir", str(attempt)],
            )
            rows = len(pd.read_csv(attempt / "compiler_case_results.csv"))
            if rows != 60:
                raise RuntimeError(f"Phase 2 repeat {repeat} expected 60 rows, got {rows}")
            self._publish(
                attempt,
                attempt.parent,
                ["compiler_case_results.csv", "compiler_summary.json", "summary_print.json"],
            )
            self.complete(stage, {"record_count": rows, "attempt": attempt.name})

    def phase3(self) -> None:
        base = yaml.safe_load(Path(self.args.phase3_config).read_text(encoding="utf-8"))
        for repeat in range(1, self.args.repeats + 1):
            stage = f"phase3_repeat_{repeat}"
            self.stage = stage
            if self.should_skip(stage):
                continue
            attempt = self._attempt_dir("phase3", repeat)
            config = json.loads(json.dumps(base))
            config["paths"]["checkpoint"] = str(attempt / "phase3.sqlite")
            config["paths"]["trace_dir"] = str(attempt / "traces")
            config["paths"]["candidate_trace_dir"] = str(attempt / "candidates")
            config_path = self.root / "configs" / f"phase3_repeat_{repeat}_{attempt.name}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            self.run_command(
                stage,
                [sys.executable, "-m", "copa.phase3.cli", "--config", str(config_path), "evaluate", "--gold", "COPA/evaluation/agent_workflow_gold.jsonl", "--output-dir", str(attempt)],
            )
            agent_rows = len(pd.read_csv(attempt / "agent_case_results.csv"))
            mode_rows = len(pd.read_csv(attempt / "mode_comparison.csv"))
            if agent_rows != 36 or mode_rows != 108:
                raise RuntimeError(
                    f"Phase 3 repeat {repeat} expected 36/108 rows, got {agent_rows}/{mode_rows}"
                )
            self._publish(
                attempt,
                attempt.parent,
                ["agent_case_results.csv", "agent_summary.json", "mode_comparison.csv", "mode_comparison_summary.json", "summary_print.json"],
            )
            self.complete(stage, {"agent_records": agent_rows, "mode_records": mode_rows, "attempt": attempt.name})

    def analysis(self) -> None:
        stage = "analysis"
        self.stage = stage
        if self.should_skip(stage):
            return
        self.run_command(
            stage,
            [sys.executable, "-m", "copa.experiments.analyze_core", "--input-dir", str(self.root)],
        )
        summary = json.loads((self.root / "analysis" / "analysis_summary.json").read_text())
        if summary["phase1_records"] != 2400 or summary["phase2_records"] != 180 or summary["phase3_mode_records"] != 324:
            raise RuntimeError(f"final analysis acceptance failed: {summary}")
        if summary["figure_png_count"] != summary["figure_pdf_count"] or summary["figure_png_count"] < 9:
            raise RuntimeError(f"figure acceptance failed: {summary}")
        self.complete(stage, summary)

    def run(self) -> None:
        self.monitor.start()
        try:
            self.preflight()
            self.tests()
            self.smoke()
            self.phase1()
            self.phase2()
            self.phase3()
            self.analysis()
            self.stage = "complete"
            self.complete("complete")
            self.record_status({"stage": "complete", "status": "success"})
        finally:
            self.monitor.stop()
            commands_path = self.root / "commands.sh"
            mode = "a" if commands_path.exists() else "w"
            with commands_path.open(mode, encoding="utf-8") as handle:
                if mode == "w":
                    handle.write("#!/usr/bin/env bash\nset -euo pipefail\n")
                handle.write(f"\n# supervisor invocation {utc_now()}\n")
                handle.write("\n".join(self.commands) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    Supervisor(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
