import unittest

from ashley.features.explanation_lexical import (
    letter_frequency_distribution,
    special_character_frequency_distribution,
    total_word_count,
    short_word_ratio,
    word_character_ratio,
    average_word_length,
    average_sentence_length_characters,
    average_sentence_length_words,
    yules_k,
    word_length_frequency_distribution,
    hapax_ratio,
    word_entropy,
    word_diversity,
    punctuation_frequency_distribution,
    FUNCTION_WORDS,
    function_word_frequency_distribution,
)
from ashley.features.common import (
    total_characters,
    alphabetic_character_ratio,
    uppercase_character_ratio,
    digit_character_ratio,
) 
from ashley.features.explanation_structural import (
    total_sentence_count,
    average_sentences_per_question,
    average_misspelled_words_per_question,
)


class TestExplanationLexicalFeatures(unittest.TestCase):

    def test_special_character_frequency(self):
        result = special_character_frequency_distribution("a(b)")
        self.assertAlmostEqual(result["("], 1 / 4)
        self.assertAlmostEqual(result[")"], 1 / 4)

    def test_total_word_count(self):
        self.assertEqual(total_word_count("Hello, world!"), 2)

    def test_short_word_ratio(self):
        self.assertAlmostEqual(short_word_ratio("I love Python"), 1 / 3)

    def test_word_character_ratio(self):
        self.assertAlmostEqual(word_character_ratio("Hi there!"), 7 / 9)

    def test_average_word_length(self):
        self.assertAlmostEqual(average_word_length("Hi there"), 3.5)

    def test_word_diversity(self):
        self.assertAlmostEqual(word_diversity("hello hello world"), 2 / 3)

    def test_average_sentence_length_characters(self):
        self.assertAlmostEqual(
            average_sentence_length_characters("Hi. Hello!"),
            3.5
        )

    def test_average_sentence_length_words(self):
        self.assertAlmostEqual(
            average_sentence_length_words("Hello world. Hi!"),
            1.5
        )
    def test_yules_k(self):
        self.assertAlmostEqual(
            yules_k("cat cat dog"),
            10000 * (5 - 3) / 9
        )

    def test_word_length_distribution(self):
        result = word_length_frequency_distribution("I like Python")

        self.assertEqual(len(result), 20)
        self.assertAlmostEqual(result[1], 1 / 3)
        self.assertAlmostEqual(result[4], 1 / 3)
        self.assertAlmostEqual(result[6], 1 / 3)

    def test_hapax_ratio(self):
        self.assertAlmostEqual(
            hapax_ratio("cat cat dog bird"),
            2 / 4
        )

    def test_word_entropy(self):
        self.assertAlmostEqual(
            word_entropy("cat dog"),
            1.0
        )

    def test_type_token_ratio(self):
        self.assertAlmostEqual(
            word_diversity("cat cat dog"),
            2 / 3
        )
    def test_explanation_total_characters(self):
        text = "Hello World!"
        self.assertEqual(total_characters(text, "explanation"), 12)


    def test_explanation_alphabetic_ratio(self):
        text = "Hello123!"
        self.assertAlmostEqual(
            alphabetic_character_ratio(text, len(text)),
            5 / 9
        )


    def test_explanation_uppercase_ratio(self):
        text = "Hello World!"
        self.assertAlmostEqual(
            uppercase_character_ratio(text, len(text)),
            2 / 12
        )


    def test_explanation_digit_ratio(self):
        text = "Hello123!"
        self.assertAlmostEqual(
            digit_character_ratio(text, len(text)),
            3 / 9
        )
    def test_punctuation_frequency_distribution(self):
        text = "Hi, Bob! Really?"

        result = punctuation_frequency_distribution(text)

        self.assertEqual(len(result), 8)
        self.assertAlmostEqual(result[","], 1 / len(text))
        self.assertAlmostEqual(result["!"], 1 / len(text))
        self.assertAlmostEqual(result["?"], 1 / len(text))
        self.assertEqual(result["."], 0.0)


    def test_punctuation_frequency_empty(self):
        result = punctuation_frequency_distribution("")

        self.assertEqual(len(result), 8)
        self.assertTrue(all(value == 0.0 for value in result.values()))


    def test_punctuation_frequency_repeated(self):
        result = punctuation_frequency_distribution("Wow!!!")

        self.assertAlmostEqual(result["!"], 3 / 6)
    def test_function_word_frequency_distribution(self):
        text = "The cat and the dog are playing."
        function_words = ["the", "and", "are"]

        result = function_word_frequency_distribution(text, function_words)

        self.assertAlmostEqual(result["the"], 2 / 7)
        self.assertAlmostEqual(result["and"], 1 / 7)
        self.assertAlmostEqual(result["are"], 1 / 7)

    def test_function_word_frequency_empty(self):
        result = function_word_frequency_distribution(
            "", ["the", "and", "are"]
        )

        self.assertEqual(
            result,
            {"the": 0.0, "and": 0.0, "are": 0.0}
        )

    def test_function_word_list(self):
        self.assertEqual(len(FUNCTION_WORDS), 150)
        self.assertEqual(len(set(FUNCTION_WORDS)), 150)


    def test_function_word_frequency_default(self):
        result = function_word_frequency_distribution(
            "The cat and the dog are playing."
        )

        self.assertEqual(len(result), 150)
        self.assertAlmostEqual(result["the"], 2 / 7)
        self.assertAlmostEqual(result["and"], 1 / 7)
        self.assertAlmostEqual(result["are"], 1 / 7)


    def test_function_word_frequency_empty_default(self):
        result = function_word_frequency_distribution("")

        self.assertEqual(len(result), 150)
        self.assertTrue(all(value == 0.0 for value in result.values()))

    def test_letter_frequency_distribution(self):
        text = "AaBbC!"

        result = letter_frequency_distribution(text)

        self.assertEqual(len(result), 26)
        self.assertAlmostEqual(result["a"], 2 / 6)
        self.assertAlmostEqual(result["b"], 2 / 6)
        self.assertAlmostEqual(result["c"], 1 / 6)
        self.assertEqual(result["d"], 0.0)


    def test_letter_frequency_empty(self):
        result = letter_frequency_distribution("")

        self.assertEqual(len(result), 26)
        self.assertTrue(all(value == 0.0 for value in result.values()))

class TestExplanationStructuralFeatures(unittest.TestCase):

    def test_total_sentence_count(self):
        self.assertEqual(
            total_sentence_count("Hello world. How are you? Fine!"),
            3
        )

    def test_average_sentences_per_question(self):
        explanations = [
            "Hello. Goodbye.",
            "One sentence!",
            ""
        ]

        self.assertAlmostEqual(
            average_sentences_per_question(explanations),
            1.0
        )

    def test_average_misspelled_words_per_question(self):
        explanations = [
            "Calcluate the avrage.",
            "Retun the result.",
            "The answer is correct."
        ]

        self.assertAlmostEqual(
            average_misspelled_words_per_question(explanations),
            1.0
        )

    def test_empty_explanations(self):
        self.assertEqual(
            average_sentences_per_question([]),
            0.0
        )
        self.assertEqual(
            average_misspelled_words_per_question([]),
            0.0
        )


if __name__ == "__main__":
    unittest.main()