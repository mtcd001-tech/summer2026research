"""Unit tests for src/features/tier2.py against hand-computed expected
values. Code-token counts were cross-checked against the actual output of
Python's tokenize module for each snippet (see comments); POS/stopword
values were cross-checked against the actually-installed NLTK resources
(not assumed) since tagger output can vary by model version.
"""
from __future__ import annotations

import math

import pytest

from src.features import tier2
from tests.fixtures import load_test_config, make_group

CONFIG = load_test_config()


def _code_group(text: str):
    return make_group(r_t="code", text=text)


def _explanation_group(text: str):
    return make_group(r_t="explanation", text=text)


# ---------------------------------------------------------------------------
# Code features
# ---------------------------------------------------------------------------

COMMENT_SNIPPET = (
    "x = 1\n"
    "y = 2\n"
    "# comment line\n"
    "if x > y:\n"
    "    print(x)\n"
    "else:\n"
    "    print(y)\n"
)


def test_comment_density():
    # 7 lines total, exactly 1 comment-only line -> 1/7
    assert tier2.comment_density(_code_group(COMMENT_SNIPPET), CONFIG) == pytest.approx(1 / 7)


IDENTIFIER_SNIPPET = "x = 1\ny = 2\nz = x + y\n"
# NAME tokens excluding reserved/builtins: x(x2: assign, use), y(x2), z(x1: assign only)
# unique=3, total=5 -> diversity=3/5=0.6; hapax (count==1)=1(z)/3=0.3333


def test_identifier_diversity():
    assert tier2.identifier_diversity(_code_group(IDENTIFIER_SNIPPET), CONFIG) == pytest.approx(0.6)


def test_identifier_reuse_hapax():
    assert tier2.identifier_reuse_hapax(_code_group(IDENTIFIER_SNIPPET), CONFIG) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Question-prompt boilerplate identifiers (scripts/investigate.py Q3)
# ---------------------------------------------------------------------------

def test_extract_question_boilerplate_identifiers_snake_case():
    prompt = "Use descriptive variable names, such as triangle_base and triangle_height."
    assert tier2._extract_question_boilerplate_identifiers(prompt) == frozenset({"triangle_base", "triangle_height"})


def test_extract_question_boilerplate_identifiers_call_like():
    prompt = "Define a Python function called <strong>greet()</strong> that prints a message."
    assert tier2._extract_question_boilerplate_identifiers(prompt) == frozenset({"greet"})


def test_extract_question_boilerplate_identifiers_assignment_line():
    prompt = "Use this starting data:\n\n\tstarting_balance = 1000\n\ttransaction_list = [1, 2]\n"
    idents = tier2._extract_question_boilerplate_identifiers(prompt)
    assert idents == frozenset({"starting_balance", "transaction_list"})


def test_extract_question_boilerplate_identifiers_excludes_builtins_and_reserved_words():
    # "print(" / "input(" are call-like but must not be treated as
    # question-required identifiers -- they're Python builtins mentioned in
    # the prompt's own instructions, not names the student must define.
    prompt = "Do not use the input() function. Use the print() function to display results."
    assert tier2._extract_question_boilerplate_identifiers(prompt) == frozenset()


def test_extract_question_boilerplate_identifiers_ignores_plain_prose():
    prompt = "Write a Python program that uses a for loop to calculate the sum of odd numbers."
    assert tier2._extract_question_boilerplate_identifiers(prompt) == frozenset()


def test_question_boilerplate_identifiers_cached_per_session_and_qid():
    # Same q_id, different session, different prompt text -> must not share
    # a cache entry keyed on q_id alone.
    tier2._question_boilerplate_cache.clear()
    a = tier2._question_boilerplate_identifiers(1, 1, "uses triangle_base only")
    b = tier2._question_boilerplate_identifiers(2, 1, "uses starting_balance only")
    assert a == frozenset({"triangle_base"})
    assert b == frozenset({"starting_balance"})


def test_identifier_diversity_no_boilerplate_excludes_question_identifiers():
    tier2._question_boilerplate_cache.clear()
    # code uses two "real" identifiers (a, b) plus the boilerplate name
    # required by the prompt (triangle_base); only a/b should count.
    code = "triangle_base = 1\na = triangle_base\nb = a\n"
    prompt = "Define a variable named triangle_base."
    group = make_group(r_t="code", text=code, prompt=prompt, session=1, q_id=2)

    with_boilerplate = tier2.identifier_diversity(group, CONFIG)
    without_boilerplate = tier2.identifier_diversity_no_boilerplate(group, CONFIG)

    # idents (incl. boilerplate): triangle_base, triangle_base, a, a, b -> 5 total, 3 unique -> 0.6
    assert with_boilerplate == pytest.approx(0.6)
    # idents excluding triangle_base: a, a, b -> 3 total, 2 unique -> 2/3
    assert without_boilerplate == pytest.approx(2 / 3)
    assert without_boilerplate != with_boilerplate


def test_identifier_reuse_hapax_no_boilerplate_excludes_question_identifiers():
    tier2._question_boilerplate_cache.clear()
    code = "triangle_base = 1\na = triangle_base\nb = a\n"
    prompt = "Define a variable named triangle_base."
    group = make_group(r_t="code", text=code, prompt=prompt, session=1, q_id=3)

    # idents excluding triangle_base: a(x2), b(x1) -> counts {a:2, b:1} -> hapax=1/2
    assert tier2.identifier_reuse_hapax_no_boilerplate(group, CONFIG) == pytest.approx(0.5)


def test_no_boilerplate_variants_degrade_to_nan_when_everything_is_boilerplate():
    tier2._question_boilerplate_cache.clear()
    code = "triangle_base = 1\n"
    prompt = "Define a variable named triangle_base."
    group = make_group(r_t="code", text=code, prompt=prompt, session=1, q_id=4)

    assert math.isnan(tier2.identifier_diversity_no_boilerplate(group, CONFIG))
    assert math.isnan(tier2.identifier_reuse_hapax_no_boilerplate(group, CONFIG))


# tokenize("a a b") -> content tokens NAME 'a','a','b' (NEWLINE/ENDMARKER excluded)
# freq: a:2, b:1 out of 3 -> entropy = -(2/3 log2 2/3 + 1/3 log2 1/3) ~= 0.918296

def test_token_entropy():
    assert tier2.token_entropy(_code_group("a a b"), CONFIG) == pytest.approx(0.9182958, rel=1e-6)


# tokenize("if x:\n    return x\n") content tokens (excl. NEWLINE/INDENT/DEDENT/ENDMARKER):
#   NAME if, NAME x, OP ':', NAME return, NAME x  -> 5 tokens
# control-flow keywords present: if, return -> 2/5 = 0.4

def test_control_flow_keyword_ratio():
    code = "if x:\n    return x\n"
    assert tier2.control_flow_keyword_ratio(_code_group(code), CONFIG) == pytest.approx(0.4)


# tokenize("print(x)\n") content tokens: NAME print, OP (, NAME x, OP ) -> 4 tokens
# 'print' followed by '(' with no def/class before it -> 1 call -> 1/4 = 0.25

def test_function_call_density_counts_call():
    assert tier2.function_call_density(_code_group("print(x)\n"), CONFIG) == pytest.approx(0.25)


# tokenize("def foo():\n    pass\n") content tokens: def, foo, (, ), :, pass -> 6 tokens
# 'foo' is preceded by 'def' -> NOT a call -> 0/6 = 0.0

def test_function_call_density_excludes_def_header():
    assert tier2.function_call_density(_code_group("def foo():\n    pass\n"), CONFIG) == pytest.approx(0.0)


# "def foo(\n    pass" has an unclosed '(' -> tokenize.TokenError -> all code
# features must degrade to NaN, not raise.

def test_malformed_code_returns_nan_not_exception():
    code = "def foo(\n    pass"
    g = _code_group(code)
    assert math.isnan(tier2.identifier_diversity(g, CONFIG))
    assert math.isnan(tier2.identifier_reuse_hapax(g, CONFIG))
    assert math.isnan(tier2.token_entropy(g, CONFIG))
    assert math.isnan(tier2.control_flow_keyword_ratio(g, CONFIG))
    assert math.isnan(tier2.function_call_density(g, CONFIG))
    # comment_density is line-based, not token-based, so it does NOT need
    # valid syntax and should still compute normally
    assert not math.isnan(tier2.comment_density(g, CONFIG))


def test_empty_code_returns_nan():
    g = _code_group("")
    assert math.isnan(tier2.identifier_diversity(g, CONFIG))
    assert math.isnan(tier2.comment_density(g, CONFIG))


# ---------------------------------------------------------------------------
# Explanation features
# ---------------------------------------------------------------------------

# word_tokenize('the cat sat on the mat the cat ran') ->
#   ['the','cat','sat','on','the','mat','the','cat','ran']  (9 tokens, already lowercase/no punctuation)
# counts: the:3, cat:2, sat:1, on:1, mat:1, ran:1 -> unique=6, total=9
PROSE = "the cat sat on the mat the cat ran"


def test_ttr():
    assert tier2.ttr(_explanation_group(PROSE), CONFIG) == pytest.approx(6 / 9)


def test_hapax_ratio():
    # hapax (count==1): sat, on, mat, ran = 4; unique = 6 -> 4/6
    assert tier2.hapax_ratio(_explanation_group(PROSE), CONFIG) == pytest.approx(4 / 6)


def test_word_entropy():
    assert tier2.word_entropy(_explanation_group(PROSE), CONFIG) == pytest.approx(2.419382, rel=1e-5)


def test_non_stopword_ratio():
    # nltk english stopwords include "the" (x3) and "on" (x1) -> 4 stopword
    # occurrences out of 9 -> non-stopword ratio = 5/9
    assert tier2.non_stopword_ratio(_explanation_group(PROSE), CONFIG) == pytest.approx(5 / 9)


# sent_tokenize splits into 2 sentences; word_tokenize (incl. trailing period)
# gives lengths [7, 9] -> mean 8, sample std sqrt(((7-8)^2+(9-8)^2)/1)=sqrt(2)
# cv = sqrt(2)/8

def test_sentence_length_cv():
    text = "The cat sat on the mat. The dog ran fast in the park today."
    assert tier2.sentence_length_cv(_explanation_group(text), CONFIG) == pytest.approx(math.sqrt(2) / 8, rel=1e-5)


def test_sentence_length_cv_nan_with_one_sentence():
    assert math.isnan(tier2.sentence_length_cv(_explanation_group("Only one sentence here."), CONFIG))


# nltk.pos_tag(word_tokenize("The quick brown fox jumps over the lazy dog.")) ==
#   DT, JJ, NN, NN, VBZ, IN, DT, JJ, NN, '.'  (10 tokens total, verified against
#   the actually-installed averaged_perceptron_tagger_eng model)
# nouns (NN*): brown, fox, dog = 3/10
# verbs (VB*): jumps = 1/10
# adjectives (JJ*): quick, lazy = 2/10
# adverbs (RB*): none = 0/10
POS_SENTENCE = "The quick brown fox jumps over the lazy dog."


def test_noun_ratio():
    assert tier2.noun_ratio(_explanation_group(POS_SENTENCE), CONFIG) == pytest.approx(0.3)


def test_verb_ratio():
    assert tier2.verb_ratio(_explanation_group(POS_SENTENCE), CONFIG) == pytest.approx(0.1)


def test_adj_ratio():
    assert tier2.adj_ratio(_explanation_group(POS_SENTENCE), CONFIG) == pytest.approx(0.2)


def test_adv_ratio():
    assert tier2.adv_ratio(_explanation_group(POS_SENTENCE), CONFIG) == pytest.approx(0.0)


def test_empty_explanation_returns_nan():
    g = _explanation_group("")
    assert math.isnan(tier2.ttr(g, CONFIG))
    assert math.isnan(tier2.non_stopword_ratio(g, CONFIG))
