"""Shared code-text utilities: line classification, tolerant tokenization,
and indentation normalization. Used by both tier2 (stylometry) and tier3
(LLM-signature) features so the exact same rules apply everywhere
comment_density / blank_line_ratio / comment_to_code_ratio are computed --
the three buckets are defined to be mutually exclusive and exhaustive.
"""
from __future__ import annotations

import io
import logging
import tokenize
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)


class LineKind(Enum):
    BLANK = auto()
    COMMENT = auto()
    CODE = auto()


def classify_line(line: str) -> LineKind:
    """BLANK if whitespace-only; COMMENT if the first non-whitespace char is
    '#'; otherwise CODE (including a code line with a trailing inline
    comment -- it's still primarily code).
    """
    stripped = line.strip()
    if stripped == "":
        return LineKind.BLANK
    if stripped.startswith("#"):
        return LineKind.COMMENT
    return LineKind.CODE


def classify_lines(text: str) -> list[LineKind]:
    if text == "":
        return []
    return [classify_line(line) for line in text.splitlines()]


def tokenize_code(code: str) -> Optional[list[tokenize.TokenInfo]]:
    """Tokenize Python source, tolerating syntax errors. Student code can be
    broken mid-edit, so this returns None (not an exception) when the code
    can't be tokenized at all, letting callers degrade to NaN for whichever
    features genuinely require valid tokens.
    """
    try:
        return list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        logger.debug("tokenize failed (code left as NaN for token-based features): %s", exc)
        return None


def normalize_indentation(code: str, unit: str = "\t") -> str:
    """Rewrite each line's leading whitespace as `unit` repeated once per
    indent LEVEL (as opposed to per literal space/tab character), so
    character-count features measure indentation *behavior* rather than
    editor tab-width configuration (e.g. 2-space vs 4-space vs tab indenters
    should not differ on this basis alone).

    Primary path uses tokenize's INDENT/DEDENT stream to get the true
    indent-level depth active at each source line. Falls back to a heuristic
    -- grouping distinct leading-whitespace prefixes by length and numbering
    them in increasing order -- when the code doesn't tokenize at all (e.g. a
    student's syntactically broken snippet still has meaningful indentation
    we want to normalize).
    """
    tokens = tokenize_code(code)
    if tokens is not None:
        try:
            return _normalize_indentation_from_tokens(code, tokens, unit)
        except Exception as exc:  # pragma: no cover - defensive, falls back below
            logger.debug("token-based indentation normalization failed: %s", exc)
    return _normalize_indentation_fallback(code, unit)


def _normalize_indentation_from_tokens(code: str, tokens: list[tokenize.TokenInfo], unit: str) -> str:
    lines = code.splitlines(keepends=True)
    depth = 0
    line_depth: dict[int, int] = {}
    for tok in tokens:
        if tok.type == tokenize.INDENT:
            depth += 1
        elif tok.type == tokenize.DEDENT:
            depth = max(0, depth - 1)
        else:
            srow = tok.start[0]
            if srow not in line_depth and 1 <= srow <= len(lines):
                line_depth[srow] = depth

    out = []
    for i, line in enumerate(lines, start=1):
        if line.strip() == "":
            out.append(line)
            continue
        stripped = line.lstrip(" \t")
        level = line_depth.get(i, 0)
        out.append(unit * level + stripped)
    return "".join(out)


def _normalize_indentation_fallback(code: str, unit: str) -> str:
    lines = code.splitlines(keepends=True)
    prefixes = set()
    for line in lines:
        if line.strip() == "":
            continue
        prefix = line[: len(line) - len(line.lstrip(" \t"))]
        if prefix:
            prefixes.add(prefix)
    ordered = sorted(prefixes, key=len)
    level_map = {p: i + 1 for i, p in enumerate(ordered)}

    out = []
    for line in lines:
        if line.strip() == "":
            out.append(line)
            continue
        prefix = line[: len(line) - len(line.lstrip(" \t"))]
        stripped = line[len(prefix) :]
        level = level_map.get(prefix, 0)
        out.append(unit * level + stripped)
    return "".join(out)
