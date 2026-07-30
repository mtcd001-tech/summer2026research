"""Unit tests for src/codetext.py: line classification and indentation
normalization (both the tokenize-based primary path and the fallback used
when code doesn't tokenize)."""
from __future__ import annotations

from src.codetext import LineKind, classify_line, classify_lines, normalize_indentation


def test_classify_line_blank():
    assert classify_line("") == LineKind.BLANK
    assert classify_line("   ") == LineKind.BLANK
    assert classify_line("\t") == LineKind.BLANK


def test_classify_line_comment():
    assert classify_line("# a comment") == LineKind.COMMENT
    assert classify_line("    # indented comment") == LineKind.COMMENT


def test_classify_line_code_including_inline_comment():
    assert classify_line("x = 1") == LineKind.CODE
    assert classify_line("x = 1  # trailing comment") == LineKind.CODE


def test_classify_lines_buckets_are_exhaustive():
    text = "x = 1\n# comment\n\ny = 2  # inline\n"
    lines = classify_lines(text)
    assert lines == [LineKind.CODE, LineKind.COMMENT, LineKind.BLANK, LineKind.CODE]


def test_normalize_indentation_tabs_and_spaces_agree():
    code_tabs = "if True:\n\tx = 1\n\ty = 2\n"
    code_spaces = "if True:\n    x = 1\n    y = 2\n"
    assert normalize_indentation(code_tabs, unit="I") == normalize_indentation(code_spaces, unit="I")
    assert normalize_indentation(code_spaces, unit="I") == "if True:\nIx = 1\nIy = 2\n"


def test_normalize_indentation_nested_levels():
    code = "if True:\n    if True:\n        x = 1\n"
    assert normalize_indentation(code, unit="I") == "if True:\nIif True:\nIIx = 1\n"


def test_normalize_indentation_blank_lines_untouched():
    code = "if True:\n    x = 1\n\n    y = 2\n"
    result = normalize_indentation(code, unit="I")
    # the blank line stays blank, not "I"-prefixed
    assert result.splitlines()[2] == ""


def test_normalize_indentation_fallback_on_unparseable_code():
    # unclosed '(' -> tokenize.TokenError -> falls back to the
    # leading-whitespace-prefix heuristic
    code = "def foo(\n    x = 1\n"
    result = normalize_indentation(code, unit="I")
    assert result == "def foo(\nIx = 1\n"
