"""Re-ask prompts (rung 3 of the enforcement ladder, ADR-002).

When the deterministic repair could not make an output valid, the model gets one more chance: the
original prompt, its own previous answer and the validation problems are handed back to it with
the request to answer again. :func:`build_reask_prompt` builds that prompt. It is a pure function
of its inputs; sending it is the ladder's job.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from slmstrict.schema import Schema, ValidationIssue

__all__ = ["build_reask_prompt"]

_TRUNCATION_MARK = "... [truncated]"


def build_reask_prompt(
    prompt: str,
    previous_output: str,
    issues: Sequence[ValidationIssue],
    *,
    schema: Schema[Any] | None = None,
    max_issues: int = 10,
    max_output_chars: int = 4000,
    max_message_chars: int = 300,
) -> str:
    """Build a correction prompt from the original ``prompt``, the model's ``previous_output`` and
    the validation ``issues`` it caused.

    The backends take a bare prompt string and keep no conversation, so the correction prompt
    carries everything: the original prompt first, then the previous answer, the problems
    (``$.path: message``, at most ``max_issues``, each cut to ``max_message_chars``), the JSON
    Schema when ``schema`` is given, and the instruction to reply with the corrected JSON only.
    ``previous_output`` is cut to ``max_output_chars`` so a runaway output cannot blow up the
    context window.
    """
    if max_issues < 1 or max_output_chars < 1 or max_message_chars < 1:
        raise ValueError("max_issues, max_output_chars and max_message_chars must be >= 1")

    lines = [
        prompt.rstrip(),
        "",
        "---",
        "Your previous answer does not satisfy the required JSON schema.",
        "",
        "Previous answer:",
        _clip(previous_output, max_output_chars),
        "",
        "Problems found:",
    ]
    shown = issues[:max_issues]
    if shown:
        lines.extend(f"- {_clip(str(issue), max_message_chars)}" for issue in shown)
        if len(issues) > len(shown):
            lines.append(f"- ... and {len(issues) - len(shown)} more")
    else:
        lines.append("- the answer is not a valid JSON object")
    if schema is not None:
        schema_text = json.dumps(schema.json_schema, separators=(",", ":"))
        lines += ["", "Required JSON schema:", schema_text]
    lines += [
        "",
        "Reply with the corrected JSON object only. "
        "Do not add explanations and do not wrap it in markdown fences.",
    ]
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + _TRUNCATION_MARK
