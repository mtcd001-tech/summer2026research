"""Tier 2: text/code stylometric features computed on the final response
text (stored response text, not a keystroke reconstruction).

Code features (`CODE_FEATURES`) use Python's own tokenize module, never
regex, and degrade to NaN (logging why) when the code doesn't tokenize --
student code can be mid-edit and syntactically broken.

Explanation features (`EXPLANATION_FEATURES`) use NLTK for tokenization,
sentence splitting, and POS tagging. NLTK was chosen (over spaCy) because it
has no separate model-download step beyond a few small corpora
(punkt/punkt_tab, averaged_perceptron_tagger_eng, stopwords) and this
project has no other NLP dependency to match. If NLTK's data files aren't
present, the functions raise LookupError with the exact `nltk.download(...)`
call needed, rather than silently returning NaN -- a missing corpus is a
setup problem, not a property of the text.
"""
from __future__ import annotations

import builtins as builtins_module
import keyword
import logging
import math
import re
import string
import tokenize
from collections import Counter

from src.codetext import LineKind, classify_lines, tokenize_code
from src.statutils import coefficient_of_variation, shannon_entropy
from src.streams import Group

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "nltk is required for tier2 explanation features. Install with "
        "`pip install nltk` (see requirements.txt)."
    ) from exc

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Code features
# --------------------------------------------------------------------------

# keyword.kwlist is the authoritative reserved-word list; builtins (print,
# len, range, ...) are a SEPARATE set and are deliberately not folded in
# here -- they belong to function_call_density, not identifier extraction or
# control_flow_keyword_ratio.
RESERVED_WORDS = set(keyword.kwlist) | set(keyword.softkwlist)
BUILTIN_NAMES = set(dir(builtins_module))

CONTROL_FLOW_KEYWORDS = RESERVED_WORDS & {
    "if", "elif", "else", "for", "while", "break", "continue", "try",
    "except", "finally", "raise", "return", "with", "yield", "match", "case",
}

# Tokens that carry no lexical content of their own (line/scope structure
# markers). Excluded from the "total tokens" denominator used by
# control_flow_keyword_ratio, function_call_density, and token_entropy.
_STRUCTURAL_TOKEN_TYPES = {
    tokenize.NEWLINE,
    tokenize.NL,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
    tokenize.ENCODING,
}


def _content_tokens(tokens: list[tokenize.TokenInfo]) -> list[tokenize.TokenInfo]:
    return [t for t in tokens if t.type not in _STRUCTURAL_TOKEN_TYPES]


def _identifiers(tokens: list[tokenize.TokenInfo]) -> list[str]:
    """NAME tokens that are neither reserved words nor builtins.

    Simplification: this counts every non-reserved, non-builtin NAME,
    including attribute names after a dot (e.g. the `y` in `x.y`) and
    keyword-argument names in calls. It does not attempt full scope/AST
    analysis to distinguish "defined identifier" from "referenced name".
    """
    return [
        t.string
        for t in tokens
        if t.type == tokenize.NAME and t.string not in RESERVED_WORDS and t.string not in BUILTIN_NAMES
    ]


def _function_calls(content_tokens: list[tokenize.TokenInfo]) -> int:
    """Count NAME tokens immediately followed by '(' that are not a def/class
    header (`def foo(` / `class Foo(` are declarations, not calls).
    """
    calls = 0
    for i, tok in enumerate(content_tokens[:-1]):
        if tok.type != tokenize.NAME or tok.string in RESERVED_WORDS:
            continue
        nxt = content_tokens[i + 1]
        if nxt.type == tokenize.OP and nxt.string == "(":
            prev = content_tokens[i - 1] if i > 0 else None
            if prev is not None and prev.type == tokenize.NAME and prev.string in ("def", "class"):
                continue
            calls += 1
    return calls


def identifier_diversity(group: Group, config: dict) -> float:
    tokens = tokenize_code(group.text or "")
    if tokens is None:
        return math.nan
    idents = _identifiers(tokens)
    if not idents:
        return math.nan
    return len(set(idents)) / len(idents)


def identifier_reuse_hapax(group: Group, config: dict) -> float:
    tokens = tokenize_code(group.text or "")
    if tokens is None:
        return math.nan
    idents = _identifiers(tokens)
    if not idents:
        return math.nan
    counts = Counter(idents)
    hapax = sum(1 for c in counts.values() if c == 1)
    return hapax / len(counts)


# --------------------------------------------------------------------------
# Question-prompt boilerplate identifiers
# --------------------------------------------------------------------------
# scripts/investigate.py's Q3 found tier2_code_* identifier features are
# substantially measuring the provided prototype: shared-identifier
# occurrence runs 14%-97% of a solution's tokens (most questions 30%-70%).
#
# Deliberately NOT stripped by "identifiers appearing in >=90% of
# participants" -- that statistical filter is circular. If AI-assisted code
# converges on similar identifier names across participants, that
# convergence is exactly the signal this project wants identifier_diversity
# to detect, and a cross-participant-frequency filter would delete it before
# it's ever measured.
#
# Instead the boilerplate set is derived from the QUESTION TEXT itself, which
# explicitly names the identifiers a correct solution is required to use
# (e.g. "Define deposit_money(current_balance, transaction_amount)...",
# "<strong>main()</strong>"). This only removes identifiers the prompt
# assigned, never identifiers participants happened to converge on
# independently.

_SNAKE_CASE_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CALL_LIKE_TOKEN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_ASSIGNMENT_LINE_RE = re.compile(r"^\s*(?:\d+\.\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)")

_question_boilerplate_cache: dict[tuple[int, int], frozenset[str]] = {}


def _extract_question_boilerplate_identifiers(prompt: str) -> frozenset[str]:
    """Identifier-shaped tokens the raw prompt text explicitly names, via
    three simple, auditable signals (HTML tags and all -- no need to strip
    them, they don't look like identifiers):

      1. snake_case tokens (>=1 underscore): virtually never occurs in
         ordinary English prose, so this is a low-false-positive way to catch
         names like "triangle_base" or "transaction_amount" wherever they
         appear in the prompt, tagged or not.
      2. a bare word immediately followed by "(": catches required function
         names however they're introduced, e.g. "<strong>main()</strong>" or
         "function called multiply_list(nums)".
      3. the left-hand side of a literal assignment line given as starter
         code, e.g. "starting_balance = 1000" or "2. width = 5".

    Reserved words and builtins are dropped -- rule 2 would otherwise catch
    the prompt's own mentions of print()/input().
    """
    found: set[str] = set()
    found.update(_SNAKE_CASE_TOKEN_RE.findall(prompt))
    found.update(m.group(1) for m in _CALL_LIKE_TOKEN_RE.finditer(prompt))
    for line in prompt.splitlines():
        m = _ASSIGNMENT_LINE_RE.match(line)
        if m:
            found.add(m.group(1))
    found -= RESERVED_WORDS
    found -= BUILTIN_NAMES
    return frozenset(found)


def _question_boilerplate_identifiers(session: int, q_id: int, prompt: str | None) -> frozenset[str]:
    """Cached per (session, q_id) -- q_id alone is not unique across sessions
    (sessions 1 and 2 share the same six prompts; sessions 3-5 each have
    their own), so both are needed to key the cache correctly."""
    key = (session, q_id)
    cached = _question_boilerplate_cache.get(key)
    if cached is not None:
        return cached
    result = _extract_question_boilerplate_identifiers(prompt or "")
    if prompt:
        _question_boilerplate_cache[key] = result
    return result


def identifier_diversity_no_boilerplate(group: Group, config: dict) -> float:
    """identifier_diversity, excluding identifiers the question prompt
    itself required (see the "Question-prompt boilerplate identifiers"
    section above). Emitted side-by-side with identifier_diversity rather
    than replacing it, so the two can be compared instead of committing
    blindly to one.
    """
    tokens = tokenize_code(group.text or "")
    if tokens is None:
        return math.nan
    boilerplate = _question_boilerplate_identifiers(group.session, group.q_id, group.prompt)
    idents = [i for i in _identifiers(tokens) if i not in boilerplate]
    if not idents:
        return math.nan
    return len(set(idents)) / len(idents)


def identifier_reuse_hapax_no_boilerplate(group: Group, config: dict) -> float:
    """identifier_reuse_hapax, excluding question-prompt boilerplate
    identifiers (see identifier_diversity_no_boilerplate)."""
    tokens = tokenize_code(group.text or "")
    if tokens is None:
        return math.nan
    boilerplate = _question_boilerplate_identifiers(group.session, group.q_id, group.prompt)
    idents = [i for i in _identifiers(tokens) if i not in boilerplate]
    if not idents:
        return math.nan
    counts = Counter(idents)
    hapax = sum(1 for c in counts.values() if c == 1)
    return hapax / len(counts)


def token_entropy(group: Group, config: dict) -> float:
    tokens = tokenize_code(group.text or "")
    if tokens is None:
        return math.nan
    content = _content_tokens(tokens)
    if not content:
        return math.nan
    counts = Counter(t.string for t in content)
    return shannon_entropy(list(counts.values()))


def comment_density(group: Group, config: dict) -> float:
    lines = classify_lines(group.text or "")
    if not lines:
        return math.nan
    comments = sum(1 for l in lines if l == LineKind.COMMENT)
    return comments / len(lines)


def control_flow_keyword_ratio(group: Group, config: dict) -> float:
    tokens = tokenize_code(group.text or "")
    if tokens is None:
        return math.nan
    content = _content_tokens(tokens)
    if not content:
        return math.nan
    cf = sum(1 for t in content if t.type == tokenize.NAME and t.string in CONTROL_FLOW_KEYWORDS)
    return cf / len(content)


def function_call_density(group: Group, config: dict) -> float:
    tokens = tokenize_code(group.text or "")
    if tokens is None:
        return math.nan
    content = _content_tokens(tokens)
    if not content:
        return math.nan
    return _function_calls(content) / len(content)


CODE_FEATURES: dict[str, callable] = {
    "identifier_diversity": identifier_diversity,
    "identifier_diversity_no_boilerplate": identifier_diversity_no_boilerplate,
    "identifier_reuse_hapax": identifier_reuse_hapax,
    "identifier_reuse_hapax_no_boilerplate": identifier_reuse_hapax_no_boilerplate,
    "token_entropy": token_entropy,
    "comment_density": comment_density,
    "control_flow_keyword_ratio": control_flow_keyword_ratio,
    "function_call_density": function_call_density,
}


# --------------------------------------------------------------------------
# Explanation (English prose) features
# --------------------------------------------------------------------------

_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
_VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
_ADJ_TAGS = {"JJ", "JJR", "JJS"}
_ADV_TAGS = {"RB", "RBR", "RBS"}

_NLTK_LOOKUP_HINT = (
    "NLTK resource not found ({resource}). Run: python -c \"import nltk; "
    "nltk.download('{package}')\""
)

_stopwords_cache: set[str] | None = None


def _stopwords() -> set[str]:
    global _stopwords_cache
    if _stopwords_cache is None:
        try:
            _stopwords_cache = set(nltk_stopwords.words("english"))
        except LookupError as exc:
            raise LookupError(_NLTK_LOOKUP_HINT.format(resource="stopwords", package="stopwords")) from exc
    return _stopwords_cache


def _tokenize_words(text: str) -> list[str]:
    try:
        return word_tokenize(text)
    except LookupError as exc:
        raise LookupError(_NLTK_LOOKUP_HINT.format(resource="punkt/punkt_tab", package="punkt_tab")) from exc


def _sentences(text: str) -> list[str]:
    try:
        return [s for s in sent_tokenize(text) if s.strip()]
    except LookupError as exc:
        raise LookupError(_NLTK_LOOKUP_HINT.format(resource="punkt/punkt_tab", package="punkt_tab")) from exc


def _normalize_words(text: str, config: dict) -> list[str]:
    """Lowercase and strip surrounding punctuation before tokenizing, per
    config.yaml's tier2.text settings. This is a deliberate, documented
    choice: without it, "Code" and "code." would count as different tokens
    from "code", which materially inflates ttr/hapax_ratio/word_entropy.
    """
    cfg = config["tier2"]["text"]
    out = []
    for w in _tokenize_words(text):
        if cfg.get("lowercase", True):
            w = w.lower()
        if cfg.get("strip_punctuation", True):
            w = w.strip(string.punctuation)
        if w:
            out.append(w)
    return out


def ttr(group: Group, config: dict) -> float:
    words = _normalize_words(group.text or "", config)
    if not words:
        return math.nan
    return len(set(words)) / len(words)


def hapax_ratio(group: Group, config: dict) -> float:
    words = _normalize_words(group.text or "", config)
    if not words:
        return math.nan
    counts = Counter(words)
    hapax = sum(1 for c in counts.values() if c == 1)
    return hapax / len(counts)


def word_entropy(group: Group, config: dict) -> float:
    words = _normalize_words(group.text or "", config)
    if not words:
        return math.nan
    counts = Counter(words)
    return shannon_entropy(list(counts.values()))


def non_stopword_ratio(group: Group, config: dict) -> float:
    words = _normalize_words(group.text or "", config)
    if not words:
        return math.nan
    stops = _stopwords()
    non_stop = sum(1 for w in words if w not in stops)
    return non_stop / len(words)


def sentence_length_cv(group: Group, config: dict) -> float:
    sentences = _sentences(group.text or "")
    if len(sentences) < 2:
        return math.nan
    lengths = [float(len(_tokenize_words(s))) for s in sentences]
    return coefficient_of_variation(lengths)


def _pos_tags(text: str) -> list[tuple[str, str]]:
    tokens = _tokenize_words(text)
    if not tokens:
        return []
    try:
        return nltk.pos_tag(tokens)
    except LookupError as exc:
        raise LookupError(
            _NLTK_LOOKUP_HINT.format(resource="averaged_perceptron_tagger_eng", package="averaged_perceptron_tagger_eng")
        ) from exc


def _pos_ratio(group: Group, tagset: set[str]) -> float:
    tags = _pos_tags(group.text or "")
    if not tags:
        return math.nan
    count = sum(1 for _, tag in tags if tag in tagset)
    return count / len(tags)


def noun_ratio(group: Group, config: dict) -> float:
    return _pos_ratio(group, _NOUN_TAGS)


def verb_ratio(group: Group, config: dict) -> float:
    return _pos_ratio(group, _VERB_TAGS)


def adj_ratio(group: Group, config: dict) -> float:
    return _pos_ratio(group, _ADJ_TAGS)


def adv_ratio(group: Group, config: dict) -> float:
    return _pos_ratio(group, _ADV_TAGS)


EXPLANATION_FEATURES: dict[str, callable] = {
    "ttr": ttr,
    "hapax_ratio": hapax_ratio,
    "word_entropy": word_entropy,
    "non_stopword_ratio": non_stopword_ratio,
    "sentence_length_cv": sentence_length_cv,
    "noun_ratio": noun_ratio,
    "verb_ratio": verb_ratio,
    "adj_ratio": adj_ratio,
    "adv_ratio": adv_ratio,
}
