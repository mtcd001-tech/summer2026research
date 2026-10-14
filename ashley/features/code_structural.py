import io
import tokenize

import re
from spellchecker import SpellChecker


def count_misspelled_comment_words(code: str) -> int:
    """Count misspelled word occurrences in Python comments."""

    spell = SpellChecker()

    # Common programming terms that should not count as misspellings
    spell.word_frequency.load_words([
        "args", "kwargs", "bool", "str", "int",
        "nums", "numpy", "pandas", "idx"
    ])

    comments = extract_comments(code)
    misspelled_count = 0

    for comment in comments:
        words = re.findall(r"[A-Za-z]+", comment)

        for word in words:
            if spell.unknown([word.lower()]):
                misspelled_count += 1

    return misspelled_count

def average_misspelled_comment_words(code_responses: list[str]) -> float:
    """Return average misspelled comment words per question."""

    if len(code_responses) == 0:
        return 0.0

    total_misspellings = sum(
        count_misspelled_comment_words(code)
        for code in code_responses
    )

    return total_misspellings / len(code_responses)

def total_line_count(code: str) -> int:
    """Return the total number of physical lines in a code response.

    Code lines, comment lines, and blank lines are all included.
    A final newline does not create an additional blank line.
    """

    return len(code.splitlines())

def average_lines_per_question(code_responses: tuple) -> float:
    """Return the average number of lines per question.

    This is calculated across all code responses for one participant.
    """

    if len(code_responses) == 0:
        return 0.0

    total_lines = sum(
        total_line_count(code)
        for code in code_responses
    )

    return total_lines / len(code_responses)

def blank_line_count(code: str) -> int:
    """Return the number of blank physical lines.

    A line containing no characters or only whitespace is considered blank.
    """

    lines = code.splitlines()

    return sum(
        line.strip() == ""
        for line in lines
    )


def blank_line_ratio(code: str) -> float:
    """Return the ratio of blank lines to total physical lines."""

    total_lines = total_line_count(code)

    if total_lines == 0:
        return 0.0

    return blank_line_count(code) / total_lines

def pure_comment_line_count(code: str) -> int:
    """Return the number of pure-comment lines.

    A pure-comment line contains only optional whitespace
    followed by a # comment.

    Lines containing code and an inline comment are not counted.
    Docstrings are not counted as comments.
    """

    return sum(
        line.lstrip().startswith("#")
        for line in code.splitlines()
    )

def comment_line_ratio(code: str) -> float:
    """Return the ratio of pure-comment lines to total physical lines."""

    total_lines = total_line_count(code)

    if total_lines == 0:
        return 0.0

    return pure_comment_line_count(code) / total_lines

def extract_comments(code: str) -> list[str]:
    """Extract Python comments, excluding strings and code."""

    comments = []

    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)

        for token in tokens:
            if token.type == tokenize.COMMENT:
                comments.append(token.string.lstrip("#").strip())

    except (tokenize.TokenError, IndentationError):
        pass

    return comments

#UNNEEDED TEMP TESTS
# if __name__ == "__main__":
#     test_code = """
# def find_highest(scores):
#     highest = scores[0]

#     for score in scores:
#         if score > highest:
#             highest = score

#     return highest
# """

#     print("Total lines:", total_line_count(test_code))
#     test_responses = (
#         "line 1\nline 2",
#         "line 1\nline 2\nline 3",
#         "line 1"
#     )

#     print(
#         "Average lines per question:",
#         average_lines_per_question(test_responses)
#     )
#     print("Blank lines:", blank_line_count(test_code))
#     print("Blank line ratio:", blank_line_ratio(test_code))
#     comment_test_code = """
# # This is a pure comment
# x = 5

#     # This is also a pure comment
# y = 10  # This is an inline comment
# print(x + y)
# """

#     print(
#         "Pure comment lines:",
#         pure_comment_line_count(comment_test_code)
#     )

#     print(
#         "Comment line ratio:",
#         comment_line_ratio(comment_test_code)
#     )