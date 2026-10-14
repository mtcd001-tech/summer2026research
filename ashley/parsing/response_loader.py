import json


def load_responses(filepath: str) -> tuple[list, int]:
    """Load response data from a JSON file.

    Args:
        filepath: Path to the JSON response file.

    Returns:
        A tuple containing:
            - the list of question responses
            - the session number
    """
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    responses = data["responses"]
    session = responses[0]["session"]

    return responses, session


def get_code_responses(responses: list) -> tuple:
    """Return the code response for each question."""

    session = responses[0]["session"]

    if session == 1:
        code_field = "code"
    elif session in (2, 5):
        code_field = "retype"
    elif session in (3, 4):
        code_field = "code_version_2"
    else:
        raise ValueError(f"Invalid session number: {session}")

    return tuple(
        response[code_field]
        for response in responses
    )


def get_explanation_responses(responses: list) -> tuple:
    """Return the explanation response for each question."""

    session = responses[0]["session"]

    if session == 1:
        explanation_field = "explanation"
    elif session in (2, 3, 4, 5):
        explanation_field = "explanation_version_2"
    else:
        raise ValueError(f"Invalid session number: {session}")

    return tuple(
        response[explanation_field]
        for response in responses
    )

#TEST
if __name__ == "__main__":
    responses, session = load_responses(
        "testdata.json"
    )

    code_responses = get_code_responses(responses)
    explanation_responses = get_explanation_responses(responses)

    print("Session:", session)
    print("Questions:", len(responses))

    print("First code response:")
    print(code_responses[0])

    print("First explanation:")
    print(explanation_responses[0])