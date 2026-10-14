import string
import re
from collections import Counter

import math

# Replace this with the exact 22 characters from your spreadsheet.
SPECIAL_CHARACTERS = "()<>%—{}[]/#̃+-*=$ˆ&"
# Features 82-89
PUNCTUATION_MARKS = ('"', ',', '.', '?', '!', ':', ';', "'")

FUNCTION_WORDS = [
    "a", "about", "above", "after", "all", "although",
    "am", "among", "an", "and", "another", "any",
    "anybody", "anyone", "anything", "are", "around", "as",
    "at", "be", "because", "before", "behind", "below",
    "beside", "between", "both", "but", "by", "can",
    "cos", "do", "down", "each", "either", "enough",
    "every", "everybody", "everyone", "everything", "few",
    "following", "for", "from", "have", "he", "her",
    "him", "i", "if", "in", "including", "inside",
    "into", "is", "it", "its", "latter", "less",
    "like", "little", "lots", "many", "me", "more",
    "most", "much", "must", "my", "near", "need",
    "neither", "no", "nobody", "none", "nor", "nothing",
    "of", "off", "on", "once", "one", "onto",
    "opposite", "or", "our", "outside", "over", "own",
    "past", "per", "plenty", "plus", "regarding", "same",
    "several", "she", "should", "since", "so", "some",
    "somebody", "someone", "something", "such", "than",
    "that", "the", "their", "them", "these", "they",
    "this", "those", "though", "through", "till", "to",
    "toward", "towards", "under", "unless", "unlike",
    "until", "up", "upon", "us", "used", "via",
    "we", "what", "whatever", "when", "where", "whether",
    "which", "while", "who", "whoever", "whom", "whose",
    "will", "with", "within", "without", "worth", "would",
    "yes", "you", "your",
]

def letter_frequency_distribution(text: str) -> dict[str, float]:
    """Return the frequency of each letter (a-z) divided by total characters."""

    total_chars = len(text)

    frequencies = {letter: 0.0 for letter in string.ascii_lowercase}

    if total_chars == 0:
        return frequencies

    for char in text.lower():
        if char in frequencies:
            frequencies[char] += 1 / total_chars

    return frequencies




# Features 32–53

#SHOULD EXCLUDE THIS FROM PRIMARY DATA SET BC 
# I think this feature may have the same problem of some of these special characters 
# being prompted by specific coding quetions-ex the arithmateic operators (%, /, //, +, -, *),  
# the # for questions specifically asking for comments, etc
#COuld still use it for transcription versus bonafide level
def special_character_frequency_distribution(text: str) -> dict[str, float]:
    """Frequency of each special character divided by total characters."""
    total = len(text)
    return {
        char: text.count(char) / total if total else 0.0
        for char in SPECIAL_CHARACTERS
    }


def extract_words(text: str) -> list[str]:
    """Extract words from explanation text, ignoring punctuation."""
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


# Feature 54
def total_word_count(text: str) -> int:
    return len(extract_words(text))


# Feature 55
def short_word_ratio(text: str) -> float:
    words = extract_words(text)
    if not words:
        return 0.0
    return sum(len(word) < 4 for word in words) / len(words)


# Feature 56
def word_character_ratio(text: str) -> float:
    words = extract_words(text)
    if not text:
        return 0.0
    return sum(len(word) for word in words) / len(text)


# Feature 57
def average_word_length(text: str) -> float:
    words = extract_words(text)
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


# Feature 60
def word_diversity(text: str) -> float:
    words = extract_words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def extract_sentences(text: str) -> list[str]:
    """Split explanation text into sentences."""
    return [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]


# Feature 58
def average_sentence_length_characters(text: str) -> float:
    sentences = extract_sentences(text)
    if not sentences:
        return 0.0
    return sum(len(sentence) for sentence in sentences) / len(sentences)


# Feature 59
def average_sentence_length_words(text: str) -> float:
    sentences = extract_sentences(text)
    if not sentences:
        return 0.0
    return sum(len(extract_words(sentence)) for sentence in sentences) / len(sentences)

#Feature 61
def yules_k(text: str) -> float:
    """Calculate Yule's K vocabulary concentration measure."""
    words = extract_words(text)
    n = len(words)

    if n == 0:
        return 0.0

    counts = Counter(words)
    frequency_squared_sum = sum(
        count ** 2 for count in counts.values()
    )

    return 10000 * (frequency_squared_sum - n) / (n ** 2)

#Feature 62-81

def word_length_frequency_distribution(text: str) -> dict[int, float]:
    """Return word-length frequencies for lengths 1 through 20."""
    words = extract_words(text)
    total_words = len(words)

    frequencies = {length: 0.0 for length in range(1, 21)}

    if total_words == 0:
        return frequencies

    for word in words:
        length = len(word)

        if 1 <= length <= 20:
            frequencies[length] += 1 / total_words

    return frequencies

#Feature 82
def hapax_ratio(text: str) -> float:
    """Return the proportion of words occurring exactly once."""
    words = extract_words(text)

    if not words:
        return 0.0

    counts = Counter(words)
    single_occurrence_count = sum(
        count == 1 for count in counts.values()
    )

    return single_occurrence_count / len(words)

#Feature 83
def word_entropy(text: str) -> float:
    """Calculate Shannon entropy of word frequencies."""
    words = extract_words(text)

    if not words:
        return 0.0

    counts = Counter(words)
    total_words = len(words)

    return -sum(
        (count / total_words) * math.log2(count / total_words)
        for count in counts.values()
    )

def word_diversity(text: str) -> float:
    words = extract_words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)
#Features 82-89
def punctuation_frequency_distribution(text: str) -> dict[str, float]:
    """Return the frequency of each punctuation mark per character."""
    total_chars = len(text)

    if total_chars == 0:
        return {mark: 0.0 for mark in PUNCTUATION_MARKS}

    return {
        mark: text.count(mark) / total_chars
        for mark in PUNCTUATION_MARKS
    }

#Feature 89-238
def function_word_frequency_distribution(
    text: str,
    function_words: list[str]
) -> dict[str, float]:
    """Calculate the frequency of each function word."""
    words = extract_words(text)
    total_words = len(words)

    if total_words == 0:
        return {word: 0.0 for word in function_words}

    counts = Counter(words)

    return {
        word: counts[word] / total_words
        for word in function_words
    }

def function_word_frequency_distribution(
    text: str,
    function_words: list[str] | None = None
) -> dict[str, float]:
    """Calculate the relative frequency of each function word."""
    if function_words is None:
        function_words = FUNCTION_WORDS

    words = extract_words(text)
    total_words = len(words)

    if total_words == 0:
        return {word: 0.0 for word in function_words}

    counts = Counter(words)

    return {
        word: counts[word] / total_words
        for word in function_words
    }