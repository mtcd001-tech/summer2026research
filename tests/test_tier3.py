"""Unit tests for src/features/tier3.py against hand-computed expected values."""
from __future__ import annotations

import math

import pytest

from src.features import tier3
from tests.fixtures import load_test_config, make_group

CONFIG = load_test_config()


def test_code_length_relative_to_prompt():
    g = make_group(r_t="code", text="ab", prompt="abcd")
    assert tier3.code_length_relative_to_prompt(g, CONFIG) == pytest.approx(0.5)


def test_code_length_relative_to_prompt_indentation_normalized():
    # 4-space and tab indentation should normalize to the SAME length before
    # the ratio is computed, since they represent the same one indent level.
    prompt = "x" * 10
    code_spaces = "if True:\n    x = 1\n"
    code_tabs = "if True:\n\tx = 1\n"
    g_spaces = make_group(r_t="code", text=code_spaces, prompt=prompt)
    g_tabs = make_group(r_t="code", text=code_tabs, prompt=prompt)
    assert tier3.code_length_relative_to_prompt(g_spaces, CONFIG) == pytest.approx(
        tier3.code_length_relative_to_prompt(g_tabs, CONFIG)
    )


def test_code_length_relative_to_prompt_nan_without_prompt():
    g = make_group(r_t="code", text="ab", prompt=None)
    assert math.isnan(tier3.code_length_relative_to_prompt(g, CONFIG))


# lines: "a"(code), ""(blank), "b"(code), ""(blank), ""(blank), "c"(code)
# 6 lines, 3 blank -> 0.5
def test_blank_line_ratio():
    text = "a\n\nb\n\n\nc\n"
    g = make_group(r_t="code", text=text)
    assert tier3.blank_line_ratio(g, CONFIG) == pytest.approx(0.5)


# lines: "# c1"(comment), "code1"(code), "# c2"(comment), "code2"(code), "code3"(code)
# comments=2, code=3 -> 2/3
def test_comment_to_code_ratio():
    text = "# c1\ncode1\n# c2\ncode2\ncode3\n"
    g = make_group(r_t="code", text=text)
    assert tier3.comment_to_code_ratio(g, CONFIG) == pytest.approx(2 / 3)


def test_comment_to_code_ratio_nan_with_no_code_lines():
    text = "# only a comment\n\n"
    g = make_group(r_t="code", text=text)
    assert math.isnan(tier3.comment_to_code_ratio(g, CONFIG))


def test_line_buckets_sum_to_total():
    text = "# c1\ncode1\n\ncode2\n"
    g = make_group(r_t="code", text=text)
    from src.codetext import LineKind, classify_lines

    lines = classify_lines(text)
    blanks = sum(1 for l in lines if l == LineKind.BLANK)
    comments = sum(1 for l in lines if l == LineKind.COMMENT)
    code = sum(1 for l in lines if l == LineKind.CODE)
    assert blanks + comments + code == len(lines)
