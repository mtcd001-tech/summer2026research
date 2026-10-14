import unittest
from pathlib import Path

from ashley.features.common import total_characters
from ashley.features.code_lexical import identifier_length_frequency_distribution, underscore_identifier_character_ratio
from ashley.features.code_structural import (
    total_line_count,
    average_lines_per_question,
    blank_line_count,
    blank_line_ratio,
    pure_comment_line_count,
    comment_line_ratio,
    extract_comments,
    count_misspelled_comment_words,
    average_misspelled_comment_words,
)
from ashley.parsing.code_parser import (
    extract_code_feature_lists,
    extract_identifier_list,
    extract_code_token_list,
)

from ashley.features.code_lexical import (
    tab_count,
    tab_character_ratio,
    total_identifier_count,
    total_keyword_count,
    short_identifier_ratio,
    short_keyword_ratio,
    identifier_character_ratio,
    keyword_character_ratio,
    average_identifier_length,
    average_keyword_length,
    double_quote_preference_ratio,
    average_line_length_characters,
    average_line_length_tokens,
    identifier_diversity,
    keyword_diversity,
)

from ashley.parsing.response_loader import (
    load_responses,
    get_code_responses,
    get_explanation_responses,
)

from ashley.features.code_complexity import (
    function_cyclomatic_complexity,
    average_cyclomatic_complexity,
)
from ashley.features.code_complexity import maximum_nesting_depth

class TestCodeStructuralFeatures(unittest.TestCase):
    def test_count_misspelled_comment_words(self):
        code = '''
    # Calcluate the avrage
    x = 5  # Retun the result
    message = "mispelled"
    '''
        self.assertEqual(count_misspelled_comment_words(code), 3)


    def test_average_misspelled_comment_words(self):
        responses = [
            "# Calcluate the avrage",
            "# Retun the result",
            "x = 5"
        ]

        self.assertEqual(
            average_misspelled_comment_words(responses),
            1.0
        )


    def test_average_misspelled_comment_words_empty(self):
        self.assertEqual(
            average_misspelled_comment_words([]),
            0.0
        )
    def test_extract_comments(self):
        code = '''
    # Calcluate the avrage
    total = 0  # Initalize total
    message = "# This is not a comment"
    '''

        self.assertEqual(
            extract_comments(code),
            ["Calcluate the avrage", "Initalize total"]
    )

    def test_total_line_count(self):
        code = "x = 1\n\ny = 2"

        self.assertEqual(total_line_count(code), 3)

    def test_average_lines_per_question(self):
        responses = (
            "line 1\nline 2",
            "line 1\nline 2\nline 3",
            "line 1",
        )

        self.assertEqual(
            average_lines_per_question(responses),
            2.0
        )

    def test_blank_line_count(self):
        code = "x = 1\n\n   \ny = 2"

        self.assertEqual(blank_line_count(code), 2)

    def test_blank_line_ratio(self):
        code = "x = 1\n\n   \ny = 2"

        self.assertAlmostEqual(
            blank_line_ratio(code),
            0.5
        )

    def test_pure_comment_line_count(self):
        code = (
            "# pure comment\n"
            "x = 5\n"
            "    # another pure comment\n"
            "y = 10  # inline comment"
        )

        self.assertEqual(
            pure_comment_line_count(code),
            2
        )

    def test_comment_line_ratio(self):
        code = (
            "# pure comment\n"
            "x = 5\n"
            "    # another pure comment\n"
            "y = 10  # inline comment"
        )

        self.assertAlmostEqual(
            comment_line_ratio(code),
            0.5
        )


class TestCommonFeatures(unittest.TestCase):

    def test_total_characters(self):
        self.assertEqual(
            total_characters("Hello123", "code"),
            8
        )

class TestCodeLexicalFeatures(unittest.TestCase):
    def setUp(self):
        self.code = """
def find_highest(scores):
    highest = scores[0]

    for score in scores:
        if score > highest:
            highest = score

    return highest
"""

        self.identifiers, self.keywords, self.tokens = (
            extract_code_feature_lists(self.code)
        )

        self.total_chars = total_characters(self.code, "code")
        self.total_tokens = len(self.tokens)

    def test_total_identifier_count(self):
        self.assertEqual(
            total_identifier_count(self.identifiers),
            11
        )

    def test_total_keyword_count(self):
        self.assertEqual(
            total_keyword_count(self.keywords),
            5
        )

    def test_short_identifier_ratio(self):
        self.assertAlmostEqual(
            short_identifier_ratio(
                self.identifiers,
                self.total_tokens
            ),
            0.0
        )

    def test_short_keyword_ratio(self):
        self.assertAlmostEqual(
            short_keyword_ratio(
                self.keywords,
                self.total_tokens
            ),
            4 / 27
        )

    def test_identifier_character_ratio(self):
        self.assertAlmostEqual(
            identifier_character_ratio(
                self.identifiers,
                self.total_chars
            ),
            73 / 129
        )

    def test_keyword_character_ratio(self):
        self.assertAlmostEqual(
            keyword_character_ratio(
                self.keywords,
                self.total_chars
            ),
            16 / 129
        )

    def test_average_identifier_length(self):
        self.assertAlmostEqual(
            average_identifier_length(self.identifiers),
            73 / 11
        )

    def test_average_keyword_length(self):
        self.assertAlmostEqual(
            average_keyword_length(self.keywords),
            16 / 5
        )

    def test_tab_count(self):
        self.assertEqual(
            tab_count(self.code),
            8
        )

    def test_tab_character_ratio(self):
        self.assertAlmostEqual(
            tab_character_ratio(
                self.code,
                self.total_chars
            ),
            8 / 129
        )

    def test_average_line_length_characters(self):
        self.assertAlmostEqual(
            average_line_length_characters(self.code),
            120 / 9
        )

    def test_average_line_length_tokens(self):
        self.assertAlmostEqual(
            average_line_length_tokens(
                self.code,
                self.total_tokens
            ),
            27 / 9
        )

    def test_identifier_diversity(self):
        self.assertAlmostEqual(
            identifier_diversity(self.identifiers),
            4 / 11
        )

    def test_keyword_diversity(self):
        self.assertAlmostEqual(
            keyword_diversity(self.keywords),
            1.0
        )

    def test_double_quote_preference_ratio(self):
        test_cases = (
        ('name = "Ashley"', 1.0),
        ("name = 'Ashley'", 0.0),
        (
            '''first = "Ashley"
second = 'Bob'
''',
                0.5
            ),
        ('''message = "Ashley's score"''', 1.0),
        ('message = f"Score: {score}"', 1.0),
        ("message = f'Score: {score}'", 0.0),            ('''message = f"""Score: {score}"""''', 1.0),
        ("x = 5", None),
        )

        for code, expected in test_cases:
            with self.subTest(code=code):
                self.assertEqual(double_quote_preference_ratio(code),expected)

    def test_identifier_length_distribution(self):
        identifiers = ["x", "total", "x", "result", "total"]
        tokens = ["x"] * 15

        result = identifier_length_frequency_distribution(
            identifiers, tokens
        )

        self.assertAlmostEqual(result[1], 2 / 15)
        self.assertAlmostEqual(result[5], 2 / 15)
        self.assertAlmostEqual(result[6], 1 / 15)
        self.assertEqual(result[3], 0.0)

    def test_identifier_length_distribution_empty(self):
        result = identifier_length_frequency_distribution([], [])

        self.assertEqual(len(result), 20)
        self.assertTrue(all(value == 0.0 for value in result.values()))

    def test_identifier_length_distribution_long_identifier(self):
        identifiers = ["x", "a" * 21]
        tokens = ["token"] * 10

        result = identifier_length_frequency_distribution(
            identifiers, tokens
        )

        self.assertAlmostEqual(result[1], 0.1)
        self.assertEqual(sum(result.values()), 0.1)

    def test_identifier_length_distribution_repeated(self):
        identifiers = ["count", "count", "count"]
        tokens = ["token"] * 6

        result = identifier_length_frequency_distribution(
            identifiers, tokens
        )

        self.assertAlmostEqual(result[5], 0.5)
    def test_identifier_length_distribution_integration(self):
        code = """
x = 5
total = x + 10
result = total * 2
"""

        identifiers = extract_identifier_list(code)
        tokens = extract_code_token_list(code)

        result = identifier_length_frequency_distribution(
            identifiers, tokens
        )

        self.assertEqual(len(result), 20)
        self.assertAlmostEqual(result[1], 2 / len(tokens))
        self.assertAlmostEqual(result[5], 2 / len(tokens))
        self.assertAlmostEqual(result[6], 1 / len(tokens))
    def test_underscore_identifier_ratio(self):
        identifiers = ["student_name", "total_score", "finalGrade"]

        result = underscore_identifier_character_ratio(identifiers)

        self.assertAlmostEqual(result, 2 / 33)


    def test_underscore_identifier_ratio_no_underscores(self):
        identifiers = ["studentName", "totalScore"]

        result = underscore_identifier_character_ratio(identifiers)

        self.assertEqual(result, 0.0)


    def test_underscore_identifier_ratio_empty(self):
        result = underscore_identifier_character_ratio([])

        self.assertEqual(result, 0.0)
class TestCodeParser(unittest.TestCase):

    def test_extract_code_feature_lists(self):
        code = """
def add(x, y):
    result = x + y
    return result
"""

        identifiers, keywords, tokens = (
            extract_code_feature_lists(code)
        )

        self.assertIsNotNone(identifiers)
        self.assertIsNotNone(keywords)
        self.assertIsNotNone(tokens)

        self.assertIn("add", identifiers)
        self.assertIn("result", identifiers)

        self.assertIn("def", keywords)
        self.assertIn("return", keywords)

        self.assertIn("+", tokens)

    def test_invalid_python_identifiers(self):
        code = "def broken("

        identifiers, keywords, tokens = (
            extract_code_feature_lists(code)
        )

        self.assertIsNone(identifiers)


class TestRealDataIntegration(unittest.TestCase):

    def setUp(self):
        self.test_file = (
            Path(__file__).parent.parent
            / "parsing"
            / "testdata.json"
        )

        self.responses, self.session = load_responses(
            self.test_file
        )

    def test_load_responses(self):
        self.assertEqual(self.session, 1)
        self.assertEqual(len(self.responses), 6)

    def test_get_code_responses(self):
        code_responses = get_code_responses(self.responses)

        self.assertEqual(len(code_responses), 6)

        for code in code_responses:
            self.assertIsInstance(code, str)

    def test_get_explanation_responses(self):
        explanations = get_explanation_responses(
            self.responses
        )

        self.assertEqual(len(explanations), 6)

        for explanation in explanations:
            self.assertIsInstance(explanation, str)

    def test_structural_features_on_real_data(self):
        code_responses = get_code_responses(self.responses)

        for code in code_responses:
            self.assertGreater(
                total_line_count(code),
                0
            )

            self.assertGreaterEqual(
                blank_line_ratio(code),
                0.0
            )
            self.assertLessEqual(
                blank_line_ratio(code),
                1.0
            )

            self.assertGreaterEqual(
                comment_line_ratio(code),
                0.0
            )
            self.assertLessEqual(
                comment_line_ratio(code),
                1.0
            )

class TestCodeComplexityFeatures(unittest.TestCase):

    def test_simple_function(self):
        code = """
def greet(name):
    print(name)
"""
        self.assertEqual(
            average_cyclomatic_complexity(code),
            1.0
        )

    def test_function_with_if(self):
        code = """
def check_number(x):
    if x > 0:
        return "positive"
    return "not positive"
"""
        self.assertEqual(
            average_cyclomatic_complexity(code),
            2.0
        )

    def test_while_with_if(self):
        code = """
def countdown(x):
    while x > 0:
        if x == 5:
            break
        x -= 1
"""
        self.assertEqual(
            average_cyclomatic_complexity(code),
            3.0
        )

    def test_multiple_functions(self):
        code = """
def foo(x):
    if x > 0:
        return x
    return 0

def bar(items):
    for item in items:
        if item > 5:
            print(item)
"""
        self.assertEqual(
            average_cyclomatic_complexity(code),
            2.5
        )

    def test_nested_functions(self):
        code = """
def outer(x):
    if x:
        print(x)

    def inner(y):
        if y:
            return y
"""
        self.assertEqual(
            average_cyclomatic_complexity(code),
            2.0
        )

    def test_invalid_python(self):
        code = "def broken("

        self.assertIsNone(
            average_cyclomatic_complexity(code)
        )

    def test_no_functions(self):
        code = """
x = 5

if x > 2:
    print(x)
"""
        self.assertIsNone(
            average_cyclomatic_complexity(code)
        )

    def test_maximum_nesting_depth_simple(self):
        code = """
def example(x):
    if x > 0:
        print(x)
"""
        self.assertEqual(maximum_nesting_depth(code), 1)

    def test_maximum_nesting_depth_nested(self):
        code = """
def example(x):
    if x > 0:
        for i in range(x):
            if i == 3:
                print(i)
"""
        self.assertEqual(maximum_nesting_depth(code), 3)

    def test_maximum_nesting_depth_separate(self):
        code = """
if x:
    print(x)

if y:
    print(y)
"""
        self.assertEqual(maximum_nesting_depth(code), 1)

    def test_maximum_nesting_depth_no_control(self):
        code = """
def example():
    print("hello")
"""
        self.assertEqual(maximum_nesting_depth(code), 0)

    def test_maximum_nesting_depth_invalid(self):
        self.assertIsNone(maximum_nesting_depth("def broken("))

    def test_maximum_nesting_depth_nested_function(self):
        code = """
def outer():
    if x:
        def inner():
            if y:
                print(y)
"""
        self.assertEqual(maximum_nesting_depth(code), 1)
if __name__ == "__main__":
    unittest.main()