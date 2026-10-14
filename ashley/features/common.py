'''Features that apply to both code and explanations
Includes: total_characters, ratio_alpha_char, ratio_upper_char, and ratio_digit_char
Includes: total_characters, alphabetic_character_ratio, uppercase_character_ratio, digit_character_ratio
'''

def total_characters(text: str, response_type: str) -> int:
    """Return the total number of characters in a response.

    For code responses, each non-overlapping group of four spaces
    counts as one character. Literal tab characters also count as
    one character.

    For explanation responses, characters are counted normally.
    """

    if response_type == "code":
        total = 0
        i = 0

        while i < len(text):
            if text[i:i + 4] == "    ":
                total += 1
                i += 4
            else:
                total += 1
                i += 1

        return total

    if response_type == "explanation":
        return len(text)

    raise ValueError(
        "response_type must be 'code' or 'explanation'"
    )

def alphabetic_character_ratio(text: str, total_chars: int) -> float:
    """Return the proportion of characters that are alphabetic."""

    if total_chars == 0:
        return 0.0

    alphabetic_count = sum(
        char.isalpha()
        for char in text
    )

    return alphabetic_count / total_chars


def uppercase_character_ratio(text: str, total_chars: int) -> float:
    """Return the proportion of characters that are uppercase."""

    if total_chars == 0:
        return 0.0

    uppercase_count = sum(
        char.isupper()
        for char in text
    )

    return uppercase_count / total_chars


def digit_character_ratio(text: str, total_chars: int) -> float:
    """Return the proportion of characters that are digits."""

    if total_chars == 0:
        return 0.0

    digit_count = sum(
        char.isdigit()
        for char in text
    )

    return digit_count / total_chars

if __name__ == "__main__":
    test_text = "Hello123"

    total = total_characters(
        test_text,
        "explanation"
    )

    print("Total characters:", total)
    print(
        "Alphabetic ratio:",
        alphabetic_character_ratio(test_text, total)
    )
    print(
        "Uppercase ratio:",
        uppercase_character_ratio(test_text, total)
    )
    print(
        "Digit ratio:",
        digit_character_ratio(test_text, total)
    )