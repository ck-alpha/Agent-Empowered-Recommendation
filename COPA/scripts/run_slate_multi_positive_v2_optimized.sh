#!/usr/bin/env bash
set -Eeuo pipefail

COPA_OPT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COPA_OPT_PACKAGE_ROOT="$(cd "${COPA_OPT_SCRIPT_DIR}/.." && pwd)"
COPA_OPT_REPOSITORY_ROOT="$(cd "${COPA_OPT_PACKAGE_ROOT}/.." && pwd)"
COPA_OPT_SOURCE_SUITE="${1:-${COPA_OPT_PACKAGE_ROOT}/results/slate_multi_positive_v2_20260719}"
COPA_OPT_RESULT_ROOT="${2:-${COPA_OPT_PACKAGE_ROOT}/results/slate_multi_positive_v2_optimized_20260721}"
COPA_OPT_WORKERS="${3:-12}"
COPA_OPT_STATUS_PATH="${COPA_OPT_RESULT_ROOT}/runner_status.json"
COPA_OPT_LOG_PATH="${COPA_OPT_RESULT_ROOT}/runner.log"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "${COPA_OPT_RESULT_ROOT}"
cd "${COPA_OPT_REPOSITORY_ROOT}"

write_status() {
  local stage="$1"
  local status="$2"
  conda run -n LLM_Rec python -c 'import datetime,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps({"stage":sys.argv[2],"status":sys.argv[3],"updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source_suite":sys.argv[4],"result_root":sys.argv[5],"workers":int(sys.argv[6]),"optimizer_kernel_version":2},indent=2,sort_keys=True)+"\n",encoding="utf-8"); t.replace(p)' "${COPA_OPT_STATUS_PATH}" "${stage}" "${status}" "${COPA_OPT_SOURCE_SUITE}" "${COPA_OPT_RESULT_ROOT}" "${COPA_OPT_WORKERS}"
}

CURRENT_STAGE="initializing"
handle_error() {
  write_status "${CURRENT_STAGE}" "failed"
}
trap handle_error ERR

run_stage() {
  CURRENT_STAGE="$1"
  shift
  write_status "${CURRENT_STAGE}" "running"
  if ! "$@" 2>&1 | tee -a "${COPA_OPT_LOG_PATH}"; then
    write_status "${CURRENT_STAGE}" "failed"
    return 1
  fi
  write_status "${CURRENT_STAGE}" "complete"
}

run_stage correctness_tests \
  conda run -n LLM_Rec pytest -q COPA/tests

run_stage validation_calibration \
  conda run -n LLM_Rec python -m copa.experiments.run_retrieval_extension \
    evaluate-suite \
    --suite-dir "${COPA_OPT_SOURCE_SUITE}" \
    --evaluation-output-dir "${COPA_OPT_RESULT_ROOT}" \
    --workers "${COPA_OPT_WORKERS}" \
    --recalibrate \
    --calibration-only

run_stage retrieval_artifact_audit \
  conda run -n LLM_Rec python -m copa.experiments.run_retrieval_extension \
    evaluate-suite \
    --suite-dir "${COPA_OPT_SOURCE_SUITE}" \
    --evaluation-output-dir "${COPA_OPT_RESULT_ROOT}" \
    --workers "${COPA_OPT_WORKERS}" \
    --resume \
    --recall-only

run_stage performance_pilot \
  conda run -n LLM_Rec python -m copa.experiments.run_retrieval_extension \
    performance-pilot \
    --suite-dir "${COPA_OPT_SOURCE_SUITE}" \
    --evaluation-output-dir "${COPA_OPT_RESULT_ROOT}" \
    --users-per-dataset 8 \
    --workers "${COPA_OPT_WORKERS}" \
    --resume

run_stage formal_evaluation \
  conda run -n LLM_Rec python -m copa.experiments.run_retrieval_extension \
    evaluate-suite \
    --suite-dir "${COPA_OPT_SOURCE_SUITE}" \
    --evaluation-output-dir "${COPA_OPT_RESULT_ROOT}" \
    --workers "${COPA_OPT_WORKERS}" \
    --resume \
    --skip-recall

CURRENT_STAGE="complete"
write_status "${CURRENT_STAGE}" "complete"
