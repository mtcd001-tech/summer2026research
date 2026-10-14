from ashley.features.common import total_characters
import io
import tokenize

def four_space_tab_count(code: str) -> int:
    """Count non-overlapping groups of four spaces as tab units."""

    count = 0
    i = 0

    while i <= len(code) - 4:
        if code[i:i + 4] == "    ":
            count += 1
            i += 4
        else:
            i += 1

    return count


def tab_count(code: str) -> int:
    """Count literal tabs and four-space tab-equivalent units."""

    literal_tabs = code.count("\t")
    space_tabs = four_space_tab_count(code)

    return literal_tabs + space_tabs


def tab_character_ratio(code: str, total_chars: int) -> float:
    """Return the ratio of tab-equivalent units to code characters."""

    if total_chars == 0:
        return 0.0

    return tab_count(code) / total_chars

def total_identifier_count(identifier_list: list | None) -> int | None:
    """Return the number of identifier occurrences."""

    if identifier_list is None:
        return None

    return len(identifier_list)


def total_keyword_count(keyword_list: list | None) -> int | None:
    """Return the number of keyword occurrences."""

    if keyword_list is None:
        return None

    return len(keyword_list)

def short_identifier_ratio(
    identifier_list: list | None,
    total_tokens: int
) -> float | None:
    """Return short identifier occurrences divided by total code tokens.

    An identifier is considered short when its length is less than 4.
    """

    if identifier_list is None:
        return None

    if total_tokens == 0:
        return 0.0

    short_count = sum(
        len(identifier) < 4
        for identifier in identifier_list
    )

    return short_count / total_tokens


def short_keyword_ratio(
    keyword_list: list | None,
    total_tokens: int
) -> float | None:
    """Return short keyword occurrences divided by total code tokens."""

    if keyword_list is None:
        return None

    if total_tokens == 0:
        return 0.0

    short_count = sum(
        len(word) < 4
        for word in keyword_list
    )

    return short_count / total_tokens

def identifier_character_ratio(
    identifier_list: list | None,
    total_chars: int
) -> float | None:
    """Return identifier characters divided by total code characters."""

    if identifier_list is None:
        return None

    if total_chars == 0:
        return 0.0

    identifier_chars = sum(
        len(identifier)
        for identifier in identifier_list
    )

    return identifier_chars / total_chars


def keyword_character_ratio(
    keyword_list: list | None,
    total_chars: int
) -> float | None:
    """Return keyword characters divided by total code characters."""

    if keyword_list is None:
        return None

    if total_chars == 0:
        return 0.0

    keyword_chars = sum(
        len(word)
        for word in keyword_list
    )

    return keyword_chars / total_chars

def average_identifier_length(
    identifier_list: list | None
) -> float | None:
    """Return the average length of identifier occurrences."""

    if identifier_list is None:
        return None

    if len(identifier_list) == 0:
        return 0.0

    return sum(
        len(identifier)
        for identifier in identifier_list
    ) / len(identifier_list)


def average_keyword_length(
    keyword_list: list | None
) -> float | None:
    """Return the average length of keyword occurrences."""

    if keyword_list is None:
        return None

    if len(keyword_list) == 0:
        return 0.0

    return sum(
        len(word)
        for word in keyword_list
    ) / len(keyword_list)

def double_quote_preference_ratio(code: str) -> float | None:
    """Return the proportion of string literals using double quotes.

    Both regular strings and f-strings are counted.

    Returns:
        A value from 0.0 to 1.0 representing double-quote preference.
        Returns None if there are no string literals or tokenization fails.
    """

    double_quoted = 0
    single_quoted = 0

    try:
        tokens = tokenize.generate_tokens(
            io.StringIO(code).readline
        )

        for token in tokens:

            # Regular string
            if token.type == tokenize.STRING:
                string_literal = token.string

                i = 0
                while (
                    i < len(string_literal)
                    and string_literal[i].lower() in {"r", "u", "b", "f"}
                ):
                    i += 1

                quote_part = string_literal[i:]

                if quote_part.startswith('"""'):
                    double_quoted += 1
                elif quote_part.startswith("'''"):
                    single_quoted += 1
                elif quote_part.startswith('"'):
                    double_quoted += 1
                elif quote_part.startswith("'"):
                    single_quoted += 1

            # F-string (Python versions with FSTRING_START)
            elif (
                hasattr(tokenize, "FSTRING_START")
                and token.type == tokenize.FSTRING_START
            ):
                fstring_start = token.string

                if '"""' in fstring_start:
                    double_quoted += 1
                elif "'''" in fstring_start:
                    single_quoted += 1
                elif '"' in fstring_start:
                    double_quoted += 1
                elif "'" in fstring_start:
                    single_quoted += 1

    except (tokenize.TokenError, IndentationError):
        return None

    total_strings = double_quoted + single_quoted

    if total_strings == 0:
        return None

    return double_quoted / total_strings

def average_line_length_characters(code: str) -> float:
    """Return the average number of characters per physical code line.

    Four-space tab equivalents count as one character.
    Interior blank lines count as lines of length zero.
    """

    lines = code.splitlines()

    if len(lines) == 0:
        return 0.0

    total_line_characters = sum(
        total_characters(line, "code")
        for line in lines
    )

    return total_line_characters / len(lines)

def average_line_length_tokens(
    code: str,
    total_tokens: int | None
) -> float | None:
    """Return the average number of tokens per physical code line."""

    if total_tokens is None:
        return None

    lines = code.splitlines()

    if len(lines) == 0:
        return 0.0

    return total_tokens / len(lines)

def identifier_diversity(
    identifier_list: list | None
) -> float | None:
    """Return distinct identifiers divided by total identifier occurrences."""

    if identifier_list is None:
        return None

    if len(identifier_list) == 0:
        return 0.0

    return len(set(identifier_list)) / len(identifier_list)


def keyword_diversity(
    keyword_list: list | None
) -> float | None:
    """Return distinct keywords divided by total keyword occurrences."""

    if keyword_list is None:
        return None

    if len(keyword_list) == 0:
        return 0.0

    return len(set(keyword_list)) / len(keyword_list)

def identifier_length_frequency_distribution(
    identifier_list: list,
    code_token_list: list
) -> dict[int, float]:
    """Return normalized identifier length frequencies (1-20)."""

    total_tokens = len(code_token_list)

    frequencies = {length: 0.0 for length in range(1, 21)}

    if total_tokens == 0:
        return frequencies

    for identifier in identifier_list:
        length = len(identifier)

        if 1 <= length <= 20:
            frequencies[length] += 1 / total_tokens

    return frequencies

def underscore_identifier_character_ratio(identifier_list: list) -> float:
    """Return the ratio of underscores to total identifier characters."""

    total_characters = sum(len(identifier) for identifier in identifier_list)

    if total_characters == 0:
        return 0.0

    underscore_count = sum(
        identifier.count("_") for identifier in identifier_list
    )

    return underscore_count / total_characters
# if __name__ == "__main__":
#     from ashley.parsing.code_parser import extract_code_feature_lists

#     test_code = """
# def find_highest(scores):
#     highest = scores[0]

#     for score in scores:
#         if score > highest:
#             highest = score

#     return highest
# """

#     identifiers, keywords, tokens = extract_code_feature_lists(test_code)

#     total_chars = total_characters(test_code, "code")
#     total_tokens = len(tokens) if tokens is not None else 0

#     print("Total characters:", total_chars)
#     print("Total tokens:", total_tokens)
#     print("Total identifiers:", total_identifier_count(identifiers))
#     print("Total keywords:", total_keyword_count(keywords))
#     print("Short identifier ratio:", short_identifier_ratio(identifiers, total_tokens))
#     print("Short keyword ratio:", short_keyword_ratio(keywords, total_tokens))
#     print("Identifier character ratio:", identifier_character_ratio(identifiers, total_chars))
#     print("Keyword character ratio:", keyword_character_ratio(keywords, total_chars))
#     print("Average identifier length:", average_identifier_length(identifiers))
#     print("Average keyword length:", average_keyword_length(keywords))
#     print("Tab count:", tab_count(test_code))
#     print("Tab character ratio:", tab_character_ratio(test_code, total_chars))
#     print("\nQuote preference tests:")

#     print(
#         double_quote_preference_ratio(
#             'name = "Ashley"'
#         )
#     )

#     print(
#         double_quote_preference_ratio(
#             "name = 'Ashley'"
#         )
#     )

#     print(
#         double_quote_preference_ratio(
#             '''first = "Ashley"
#     second = 'Bob'
#     '''
#         )
#     )

#     print(
#         double_quote_preference_ratio(
#             '''message = "Ashley's score"'''
#         )
#     )

#     print(
#         double_quote_preference_ratio(
#             'message = f"Score: {score}"'
#         )
#     )

#     print(
#         double_quote_preference_ratio(
#             "x = 5"
#         )
#     )
#     print(
#     double_quote_preference_ratio(
#         'message = f"Score: {score}"'
#     )
#     )

#     print(
#         double_quote_preference_ratio(
#             "message = f'Score: {score}'"
#         )
#     )

#     print(
#         double_quote_preference_ratio(
#             '''message = f"""Score: {score}"""'''
#         )
#     )
#     print("\nFeatures 62-65:")

#     print(
#         "Average line length (characters):",
#         average_line_length_characters(test_code)
#     )

#     print(
#         "Average line length (tokens):",
#         average_line_length_tokens(
#             test_code,
#             total_tokens
#         )
#     )

#     print(
#         "Identifier diversity:",
#         identifier_diversity(identifiers)
#     )

#     print(
#         "Keyword diversity:",
#         keyword_diversity(keywords)
#     )