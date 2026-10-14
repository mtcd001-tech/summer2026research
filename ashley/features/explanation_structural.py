from ashley.features.explanation_lexical import extract_sentences
from spellchecker import SpellChecker
import re

# Feature 239
def total_sentence_count(text: str) -> int:
    """Return the number of sentences in an explanation."""
    return len(extract_sentences(text))


# Feature 240
def average_sentences_per_question(explanations: list[str]) -> float:
    """Return the average number of sentences per question."""
    if not explanations:
        return 0.0

    total_sentences = sum(
        total_sentence_count(text) for text in explanations
    )

    return total_sentences / len(explanations)


# Feature 241
def average_misspelled_words_per_question(
    explanations: list[str]
) -> float:
    """Return average misspelled word occurrences per question."""
    if not explanations:
        return 0.0

    spell = SpellChecker()
    total_misspellings = 0

    for text in explanations:
        words = re.findall(r"[A-Za-z]+", text)

        for word in words:
            if spell.unknown([word.lower()]):
                total_misspellings += 1

    return total_misspellings / len(explanations)