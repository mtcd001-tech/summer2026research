import unittest
from pathlib import Path

from ashley.parsing.response_loader import (
    load_responses,
    get_explanation_responses,
)

from ashley.features.explanation_lexical import (
    total_word_count,
    letter_frequency_distribution,
    word_length_frequency_distribution,
    punctuation_frequency_distribution,
    function_word_frequency_distribution,
)

from ashley.features.explanation_structural import (
    total_sentence_count,
)


class TestExplanationIntegration(unittest.TestCase):

    def test_real_explanation_features(self):
        filepath = Path(__file__).resolve().parents[1] / "parsing" / "testdata.json"
        data = load_responses(str(filepath))
        responses = data[0]
        explanations = get_explanation_responses(responses)

        self.assertGreater(len(explanations), 0)

        for text in explanations[:10]:
            self.assertIsInstance(text, str)

            features = {
                "total_words": total_word_count(text),
                "letters": letter_frequency_distribution(text),
                "word_lengths": word_length_frequency_distribution(text),
                "punctuation": punctuation_frequency_distribution(text),
                "function_words": function_word_frequency_distribution(text),
                "sentences": total_sentence_count(text),
            }

            self.assertGreaterEqual(features["total_words"], 0)
            self.assertEqual(len(features["letters"]), 26)
            self.assertEqual(len(features["word_lengths"]), 20)
            self.assertEqual(len(features["punctuation"]), 8)
            self.assertEqual(len(features["function_words"]), 150)
            self.assertGreaterEqual(features["sentences"], 0)