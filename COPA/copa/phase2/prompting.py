"""Versioned prompt rendering for the constraint compiler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


class ConstraintCompilerPrompt:
    system_prompt = (
        "You are a constrained semantic compiler. Follow the supplied closed-world schema and return only valid JSON."
    )

    def __init__(self, template_path: Optional[Path] = None, *, version: str = "constraint_compiler_v2"):
        self.version = version
        path = template_path or Path(__file__).with_name("prompts") / f"{version}.txt"
        self.template = path.read_text(encoding="utf-8")

    def render(
        self,
        *,
        user_request: str,
        domain_schema: Mapping[str, Any],
        json_schema: Mapping[str, Any],
        previous_output: Optional[str] = None,
        validation_error: Optional[str] = None,
        clarification_answers: Sequence[str] = (),
    ) -> str:
        repair_context = ""
        if validation_error is not None:
            repair_context = (
                "<repair_request>\n"
                "The previous output failed validation. Correct it without changing the user's meaning.\n"
                f"Validation error: {validation_error[:2000]}\n"
                f"Previous output: {(previous_output or '')[:4000]}\n"
                "</repair_request>"
            )
        return self.template.format(
            domain_schema=json.dumps(domain_schema, ensure_ascii=False, sort_keys=True),
            json_schema=json.dumps(json_schema, ensure_ascii=False, sort_keys=True),
            user_request=user_request,
            clarification_context=json.dumps(
                [str(answer) for answer in clarification_answers],
                ensure_ascii=False,
            ),
            repair_context=repair_context,
        )
