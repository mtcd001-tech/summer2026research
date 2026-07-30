"""Tier 3: LLM-signature features computed on the final response text.

All three features apply to the code stream (r_t == "code"): they compare
code volume/shape against the prompt or against itself, which is only
meaningful for code, not prose explanations.
"""
from __future__ import annotations

import math

from src.codetext import LineKind, classify_lines, normalize_indentation
from src.streams import Group


def code_length_relative_to_prompt(group: Group, config: dict) -> float:
    """len(code) / len(prompt), using the question text matching this
    group's q_id. Code is indentation-normalized first (config.yaml
    tier2.indentation.canonical_unit) so this measures authored volume
    rather than incidental editor tab-width.
    """
    code = group.text or ""
    prompt = group.prompt or ""
    if not code or not prompt:
        return math.nan
    unit = config["tier2"]["indentation"]["canonical_unit"]
    normalized_code = normalize_indentation(code, unit=unit)
    return len(normalized_code) / len(prompt)


def blank_line_ratio(group: Group, config: dict) -> float:
    lines = classify_lines(group.text or "")
    if not lines:
        return math.nan
    blanks = sum(1 for l in lines if l == LineKind.BLANK)
    return blanks / len(lines)


def comment_to_code_ratio(group: Group, config: dict) -> float:
    lines = classify_lines(group.text or "")
    if not lines:
        return math.nan
    comments = sum(1 for l in lines if l == LineKind.COMMENT)
    code_lines = sum(1 for l in lines if l == LineKind.CODE)
    if code_lines == 0:
        return math.nan
    return comments / code_lines


FEATURES: dict[str, callable] = {
    "code_length_relative_to_prompt": code_length_relative_to_prompt,
    "blank_line_ratio": blank_line_ratio,
    "comment_to_code_ratio": comment_to_code_ratio,
}
