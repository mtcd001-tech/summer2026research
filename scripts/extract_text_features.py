import json
import ast
import io
import keyword
import tokenize
import csv
import re
from pathlib import Path

#Non-overlapping sequences of four spaces were treated as tab-equivalent indentation units.
#Literal tab characters, when present, were also counted as one tab unit.

# Helper Functions

# # File Path: C:\Users\ashle\OneDrive\Desktop\Work Info\Biometrics Research\Data Analysis and User Files\Dataset
# def open_file(filepath: str) -> tuple:
#     """Opens file to save the responses list
#     Return:
#         resp_list: List of the 6 responses for each question
#     """
#     with open(filepath, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     resp_list = data["responses"]
#     session = resp_list[0]["session"]
#     return resp_list, session

# def each_q_code(resp_list: list) -> tuple:
#     """Returns the code response for each question."""

#     session = resp_list[0]["session"]

#     if session == 1:
#         code_field = "code"

#     elif session in (2, 5):
#         code_field = "retype"

#     elif session in (3, 4):
#         code_field = "code_version_2"

#     else:
#         raise ValueError(f"Invalid session number: {session}")

#     code_tuple = tuple(
#         response[code_field] for response in resp_list
#     )

#     return code_tuple

# def each_q_exp(resp_list: list) -> tuple:
#     session = resp_list[0]["session"]
#     """For each question saves the explanation responses
#     Return:
#         q1_exp, q2_exp, q3_exp, q4_exp, q5_exp, q6_exp: 6 str explanation responses for each question
#     """
#     if session == 1:
#         q1_exp, q2_exp, q3_exp, q4_exp, q5_exp, q6_exp = (
#             response["explanation"] for response in resp_list
#         )
#     elif session >1 and session <= 5:
#         q1_exp, q2_exp, q3_exp, q4_exp, q5_exp, q6_exp = (
#             response["explanation_version_2"] for response in resp_list
#         )
#     else:
#         return None
#     exp_tuple = q1_exp, q2_exp, q3_exp, q4_exp, q5_exp, q6_exp
#     return exp_tuple

# def tab_count_four_space(ind_q_resp: str) -> int: 
#     """Returns the number of tab characters (4 spaces) 
#     Parameters:
#         ind_q_resp: str, one question's response of either code or explanation
#     Return:
#         tab_count: int, number of tab characters
#     """
#     tab_count = 0
#     i = 0
#     while i <= len(ind_q_resp) - 4:
#         if ind_q_resp[i:i+4] == "    ":
#             tab_count += 1
#             i += 4 # Skip these four spaces
#         else:
#             i += 1
#     return tab_count

# def tab_count(ind_q_resp: str) -> int:
#     """Counts literal tabs and four-space tab equivalents.
#     Parameters:
#         ind_q_resp: str, one question's response of either code or explanation
#     Return: int, number of tab characters
#     """
#     literal_tabs = ind_q_resp.count("\t")
#     space_tabs = tab_count_four_space(ind_q_resp)

#     return literal_tabs + space_tabs

#Start of individual feature extraction

# def tot_char(ind_q_resp: str, response_type: str) -> int:
#     """Returns the total characters in the question response.
#     For code, four spaces are treated as one tab character.
#     For explanations, characters are counted exactly as stored.
#     Parameters:
#         ind_q_resp: str, one question's response of either code or explanation
#         response_type: str, either "code" or "explanation"
#     Return:
#         total_char: int, total number of characters in string response
#     """
#     total_char = len(ind_q_resp)
#     if response_type == "code":
#         tab_space = tab_count_four_space(ind_q_resp)
#         total_char -= 3 * tab_space
#         #Four spaces count as one character.
#         #A literal \t is already counted as one character.
#     elif response_type == "explanation":
#         total_char = len(ind_q_resp)
#     else:
#         raise ValueError("response_type must be either 'code' or 'explanation'")
#     return total_char

# def ratio_alpha_char(ind_q_resp: str, response_type: str) -> float:
#     """Returns the ratio of alphabetic characters to total characters
#     Parameters:
#         ind_q_resp: str, one question's response of either code or explanation
#         response_type: str, either "code" or "explanation"
#     Return:
#         alpha_ratio: float, ratio of alphabetic characters to total characters
#     """
#     total_char = tot_char(ind_q_resp, response_type)
#     alpha_count = 0
#     for char in ind_q_resp:
#         if char.isalpha():
#             alpha_count += 1
#     if total_char != 0:
#         alpha_ratio = alpha_count/total_char
#     else:
#         alpha_ratio = 0
#     return alpha_ratio

# def ratio_upper_char(ind_q_resp: str, response_type: str) -> float:
#     """Returns the ratio of uppercase alphabetic characters to total characters
#     Parameters:
#         ind_q_resp: str, one question's response of either code or explanation
#         response_type: str, either "code" or "explanation"
#     Return:
#         upper_ratio: float, ratio of uppercase alphabetic characters to total characters
#     """
#     total_char = tot_char(ind_q_resp, response_type)
#     upper_count = 0
#     for char in ind_q_resp:
#         if char.isupper():
#             upper_count += 1
#     if total_char != 0:
#         upper_ratio = upper_count/total_char
#     else:
#         upper_ratio = 0
#     return upper_ratio

# def ratio_digit_char(ind_q_resp: str, response_type: str) -> float:
#     """Returns the ratio of digit characters to total characters
#     Parameters:
#         ind_q_resp: str, one question's response of either code or explanation
#         response_type: str, either "code" or "explanation"
#     Return:
#         digit_ratio: float, ratio of digit characters to total characters
#     """
#     total_char = tot_char(ind_q_resp, response_type)
#     digit_count = 0
#     for char in ind_q_resp:
#         if char.isdigit():
#             digit_count += 1
#     if total_char != 0:
#         digit_ratio = digit_count/total_char
#     else:
#         digit_ratio = 0
#     return digit_ratio

# CODE ONLY FEATURES

# def ratio_tab_char(ind_q_resp: str) -> float:
#     """Returns the ratio of tab characters to total code characters
#     Parameters:
#         ind_q_resp: str, one question's code response
#     Return:
#         tab_ratio: float, ratio of tab characters to total characters
#     """
#     total_char = tot_char(ind_q_resp, "code")
#     tab_number = tab_count(ind_q_resp) 
#     if total_char != 0:
#         tab_ratio = tab_number / total_char
#     else:
#         tab_ratio = 0
#     return tab_ratio

#QUTOE RATIO MIGRATED AND CLEANED UP
# def double_quote_preference_ratio(ind_q_resp: str) -> float | None: #PROBLEM: counts raw ' and " characters anywhere in the code, including comments., fine for prelim analysis
#     """Returns double quotes divided by all single and double quotes."""
#     double_quotes = ind_q_resp.count('"')
#     single_quotes = ind_q_resp.count("'")
#     total_quotes = double_quotes + single_quotes

#     if total_quotes == 0:
#         return None

#     return double_quotes / total_quotes

#Identifier Extraction Class and feature functions

# class IdentifierCollector(ast.NodeVisitor):
#     """Finds names that are defined by the programmer.
#     Includes: variables, function names, classes, function parameters,
#         imported local names, exception variables, attributes, and methods"""
#     def __init__(self):
#         """Creates sets for ordinary names and user-defined attributes."""
#         self.user_defined_names = set()
#         self.user_defined_attributes = set()
        
#     def visit_Name(self, node):
#         """Collects variable names when they are assigned a value.
#         Examples:
#             total = 0
#             for number in numbers:
#         """
#         if isinstance(node.ctx, ast.Store):
#             self.user_defined_names.add(node.id)
#         self.generic_visit(node)
        
#     def visit_FunctionDef(self, node):
#         """Collects names of user-defined functions."""
#         self.user_defined_names.add(node.name)
#         self.generic_visit(node)
        
#     def visit_AsyncFunctionDef(self, node):
#         """Collects names of user-defined asynchronous functions."""
#         self.user_defined_names.add(node.name)
#         self.generic_visit(node)
        
#     def visit_ClassDef(self, node):
#         """Collects user-defined class and method names."""
#         self.user_defined_names.add(node.name)
#         # Methods are attributes when called through an object,
#         # such as account.deposit().
#         for statement in node.body:
#             if isinstance(
#                 statement,
#                 (ast.FunctionDef, ast.AsyncFunctionDef)
#             ):
#                 self.user_defined_attributes.add(statement.name)
#         self.generic_visit(node)
        
#     def visit_arg(self, node):
#         """Collects function parameter names."""
#         self.user_defined_names.add(node.arg)
#         self.generic_visit(node)
        
#     def visit_alias(self, node):
#         """Collects names introduced through imports.
#         Examples:
#             import math
#             import numpy as np
#             from random import randint
#         """
#         if node.asname is not None:
#             imported_name = node.asname
#         else:
#             imported_name = node.name.split(".")[0]
#         if imported_name != "*":
#             self.user_defined_names.add(imported_name)
#         self.generic_visit(node)
        
#     def visit_ExceptHandler(self, node):
#         """Collects exception-variable names.
#         Example:
#             except ValueError as error:
#         """
#         if node.name is not None:
#             self.user_defined_names.add(node.name)
#         self.generic_visit(node)
        
#     def visit_Attribute(self, node):
#         """Collects attributes assigned by the programmer.
#         Examples:
#             self.balance = 100
#             account.owner = "Sam"
#         """
#         if isinstance(node.ctx, ast.Store):
#             self.user_defined_attributes.add(node.attr)
#         self.generic_visit(node)

# Identifier Functions

# def get_imported_name(node: ast.alias) -> str:
#     """Returns the local name introduced by an import.

#     Parameters:
#         node: ast.alias, one imported module or imported object

#     Return:
#         imported_name: str, name available inside the program
#     """
#     if node.asname is not None:
#         return node.asname
#     return node.name.split(".")[0]

# def extract_identifier_list(ind_q_resp: str) -> list | None:
#     """Returns every occurrence of a user-defined identifier.
#     Repeated occurrences are included repeatedly because several features
#     require the total number of identifier occurrences.
#     Parameters:
#         ind_q_resp: str, one question's code response
#     Return:
#         identifier_list: list, all user-defined identifier occurrences
#         None: if the code contains invalid Python syntax
#     """
#     try:
#         tree = ast.parse(ind_q_resp)
#     except SyntaxError:
#         return None
#     collector = IdentifierCollector()
#     collector.visit(tree)
#     defined_names = collector.user_defined_names
#     defined_attributes = collector.user_defined_attributes
#     identifier_list = []
#     for node in ast.walk(tree):

#         # Variable definitions and uses
#         if isinstance(node, ast.Name):
#             if node.id in defined_names:
#                 identifier_list.append(node.id)

#         # Function names where they are declared
#         elif isinstance(
#             node,
#             (ast.FunctionDef, ast.AsyncFunctionDef)
#         ):
#             if node.name in defined_names:
#                 identifier_list.append(node.name)

#         # Class names where they are declared
#         elif isinstance(node, ast.ClassDef):
#             if node.name in defined_names:
#                 identifier_list.append(node.name)

#         # Parameter names where they are declared
#         elif isinstance(node, ast.arg):
#             if node.arg in defined_names:
#                 identifier_list.append(node.arg)

#         # Imported names where they are introduced
#         elif isinstance(node, ast.alias):
#             imported_name = get_imported_name(node)

#             if imported_name in defined_names:
#                 identifier_list.append(imported_name)

#         # Exception-variable declarations
#         elif isinstance(node, ast.ExceptHandler):
#             if (
#                 node.name is not None
#                 and node.name in defined_names
#             ):
#                 identifier_list.append(node.name)

#         # User-defined attributes and method uses
#         elif isinstance(node, ast.Attribute):
#             if node.attr in defined_attributes:
#                 identifier_list.append(node.attr)

#     return identifier_list

# def extract_keyword_list(ind_q_resp: str) -> list | None:
#     """Returns every occurrence of a Python keyword.

#     Keywords inside comments and string literals are not counted.

#     Parameters:
#         ind_q_resp: str, one question's code response

#     Return:
#         keyword_list: list, all Python keyword occurrences
#         None: if the response cannot be completely tokenized
#     """
#     keyword_list = []

#     try:
#         tokens = tokenize.generate_tokens(
#             io.StringIO(ind_q_resp).readline
#         )

#         for token in tokens:
#             if (
#                 token.type == tokenize.NAME
#                 and keyword.iskeyword(token.string)
#             ):
#                 keyword_list.append(token.string)

#     except (tokenize.TokenError, IndentationError):
#         return None

#     return keyword_list

# def extract_code_token_list(ind_q_resp: str) -> list | None: #SHOULD I KEEP COMMENTS EXLUDED
#     """Returns the meaningful lexical tokens in a code response.

#     Comments, indentation markers, line-break markers, and the end marker
#     are excluded. Names, keywords, literals, operators, and punctuation
#     are included.

#     Parameters:
#         ind_q_resp: str, one question's code response

#     Return:
#         token_list: list, meaningful code-token occurrences
#         None: if the response cannot be completely tokenized
#     """
#     ignored_token_types = {
#         tokenize.COMMENT,
#         tokenize.NL,
#         tokenize.NEWLINE,
#         tokenize.INDENT,
#         tokenize.DEDENT,
#         tokenize.ENDMARKER
#     }

#     token_list = []
#     try:
#         tokens = tokenize.generate_tokens(
#             io.StringIO(ind_q_resp).readline
#         )
#         for token in tokens:
#             if (
#                 token.type not in ignored_token_types
#                 and token.string.strip() != ""
#             ):
#                 token_list.append(token.string)
#     except (tokenize.TokenError, IndentationError):
#         return None
#     return token_list

# def extract_code_feature_lists(ind_q_resp: str) -> tuple:
#     """Extracts the three reusable lists needed for features 54-65.
#     Parameters:
#         ind_q_resp: str, one question's code response
#     Return:
#         identifier_list: list or None
#         keyword_list: list or None
#         token_list: list or None
#     """
#     identifier_list = extract_identifier_list(ind_q_resp)
#     keyword_list = extract_keyword_list(ind_q_resp)
#     token_list = extract_code_token_list(ind_q_resp)

#     return identifier_list, keyword_list, token_list

# def total_identifier_count(identifier_list: list | None) -> int | None:
#     """Feature 54: Returns total identifier occurrences.

#     Return:
#         int: total identifier occurrences
#         None: if identifiers could not be extracted
#     """
#     if identifier_list is None:
#         return None

#     return len(identifier_list)
    
# def total_keyword_count(keyword_list: list | None) -> int | None:
#     """Feature 55: Returns total keyword occurrences.
#     Return:
#         int: total keyword occurrences
#         None: if keywords could not be extracted
#     """
#     if keyword_list is None:
#         return None

#     return len(keyword_list)

# def ratio_short_identifiers(identifier_list: list | None,total_tokens: int | None) -> float | None:
#     """Feature 56: Returns short identifier occurrences divided by M.

#     An identifier is short when it contains fewer than four characters.

#     Parameters:
#         identifier_list: list, identifier occurrences
#         total_tokens: int, total meaningful code tokens M

#     Return:
#         float: short identifier count divided by M
#         None: if identifiers or tokens could not be extracted
#     """
#     if identifier_list is None or total_tokens is None:
#         return None

#     if total_tokens == 0:
#         return 0.0

#     short_identifier_count = 0

#     for identifier in identifier_list:
#         if len(identifier) < 4:
#             short_identifier_count += 1

#     return short_identifier_count / total_tokens

# def ratio_short_keywords(keyword_list: list | None,total_tokens: int | None) -> float | None:
#     """Feature 57: Returns short keyword occurrences divided by M.

#     A keyword is short when it contains fewer than four characters.

#     Parameters:
#         keyword_list: list, keyword occurrences
#         total_tokens: int, total meaningful code tokens M

#     Return:
#         float: short keyword count divided by M
#         None: if keywords or tokens could not be extracted
#     """
#     if keyword_list is None or total_tokens is None:
#         return None
#     if total_tokens == 0:
#         return 0.0
#     short_keyword_count = 0
#     for keyword_name in keyword_list:
#         if len(keyword_name) < 4:
#             short_keyword_count += 1
#     return short_keyword_count / total_tokens

# def ratio_identifier_characters(identifier_list: list | None,total_characters: int) -> float | None:
#     """Feature 58: Returns identifier characters divided by C.
#     Repeated identifier occurrences contribute their characters repeatedly.
#     Parameters:
#         identifier_list: list, identifier occurrences
#         total_characters: int, adjusted code-character count C
#     Return:
#         float: identifier characters divided by C
#         None: if identifiers could not be extracted
#     """
#     if identifier_list is None:
#         return None

#     if total_characters == 0:
#         return 0.0

#     identifier_characters = 0

#     for identifier in identifier_list:
#         identifier_characters += len(identifier)

#     return identifier_characters / total_characters

# def ratio_keyword_characters(keyword_list: list | None,total_characters: int) -> float | None:
#     """Feature 59: Returns keyword characters divided by C.

#     Repeated keyword occurrences contribute their characters repeatedly.

#     Parameters:
#         keyword_list: list, keyword occurrences
#         total_characters: int, adjusted code-character count C

#     Return:
#         float: keyword characters divided by C
#         None: if keywords could not be extracted
#     """
#     if keyword_list is None:
#         return None

#     if total_characters == 0:
#         return 0.0

#     keyword_characters = 0

#     for keyword_name in keyword_list:
#         keyword_characters += len(keyword_name)

#     return keyword_characters / total_characters

# def avg_identifier_length(identifier_list: list | None) -> float | None:
#     """Feature 60: Returns average identifier length in characters.

#     Return:
#         float: average identifier length
#         None: if identifiers could not be extracted
#     """
#     if identifier_list is None:
#         return None

#     if len(identifier_list) == 0:
#         return 0.0

#     total_length = 0

#     for identifier in identifier_list:
#         total_length += len(identifier)

#     return total_length / len(identifier_list)

# def avg_keyword_length(keyword_list: list | None) -> float | None:
#     """Feature 61: Returns average keyword length in characters.

#     Return:
#         float: average keyword length
#         None: if keywords could not be extracted
#     """
#     if keyword_list is None:
#         return None

#     if len(keyword_list) == 0:
#         return 0.0

#     total_length = 0

#     for keyword_name in keyword_list:
#         total_length += len(keyword_name)

#     return total_length / len(keyword_list)

# def avg_line_length_characters(ind_q_resp: str) -> float:
#     """Feature 62: Returns average code-line length in characters.

#     Four-space tab equivalents are treated as one character through the
#     existing tot_char() function. Newline characters are not included in
#     individual line lengths. Interior blank lines count as lines of length 0.

#     Parameters:
#         ind_q_resp: str, one question's code response

#     Return:
#         float: average characters per line
#     """
#     code_lines = ind_q_resp.splitlines()
#     if len(code_lines) == 0:
#         return 0.0
#     total_line_characters = 0
#     for line in code_lines:
#         total_line_characters += tot_char(line, "code")

#     return total_line_characters / len(code_lines)

# def avg_line_length_tokens(ind_q_resp: str,total_tokens: int | None) -> float | None:
#     """Feature 63: Returns average number of tokens per code line.
#     Parameters:
#         ind_q_resp: str, one question's code response
#         total_tokens: int, total meaningful code tokens M
#     Return:
#         float: average tokens per physical line
#         None: if tokens could not be extracted
#     """
#     if total_tokens is None:
#         return None

#     code_lines = ind_q_resp.splitlines()

#     if len(code_lines) == 0:
#         return 0.0

#     return total_tokens / len(code_lines)

# def ratio_different_identifiers(identifier_list: list | None) -> float | None:
#     """Feature 64: Returns distinct identifiers divided by total identifiers.

#     Parameters:
#         identifier_list: list, all identifier occurrences

#     Return:
#         float: distinct identifier ratio
#         None: if identifiers could not be extracted
#     """
#     if identifier_list is None:
#         return None

#     total_identifiers = len(identifier_list)

#     if total_identifiers == 0:
#         return 0.0

#     different_identifiers = len(set(identifier_list))

#     return different_identifiers / total_identifiers

# def ratio_different_keywords(
#     keyword_list: list | None
# ) -> float | None:
#     """Feature 65: Returns distinct keywords divided by total keywords.

#     Parameters:
#         keyword_list: list, all keyword occurrences

#     Return:
#         float: distinct keyword ratio
#         None: if keywords could not be extracted
#     """
#     if keyword_list is None:
#         return None

#     total_keywords = len(keyword_list)

#     if total_keywords == 0:
#         return 0.0

#     different_keywords = len(set(keyword_list))

#     return different_keywords / total_keywords

#STRUCTURAL FUNCTIONS

# def total_line_count(ind_q_resp: str) -> int:
#     """Feature 241: Returns the total number of physical lines.

#     Code lines, pure-comment lines, and blank lines are all included.
#     A final newline character does not create an additional blank line.

#     Parameters:
#         ind_q_resp: str, one question's code response

#     Return:
#         total_lines: int, total number of physical lines
#     """
#     code_lines = ind_q_resp.splitlines()
#     total_lines = len(code_lines)

#     return total_lines


# def avg_lines_per_question(code_tuple: tuple) -> float:
#     """Feature 242: Returns the average number of lines per question.

#     This feature is calculated once across all code responses completed
#     by one participant.

#     Parameters:
#         code_tuple: tuple, code responses for all questions

#     Return:
#         average_lines: float, average number of lines per question
#     """
#     number_of_questions = len(code_tuple)

#     if number_of_questions == 0:
#         return 0.0

#     total_lines_all_questions = 0

#     for ind_q_resp in code_tuple:
#         total_lines_all_questions += total_line_count(ind_q_resp)

#     average_lines = (
#         total_lines_all_questions / number_of_questions
#     )

#     return average_lines


# def blank_line_count(ind_q_resp: str) -> int:
#     """Returns the number of blank physical lines.

#     A line containing no characters or only whitespace is considered blank.

#     Parameters:
#         ind_q_resp: str, one question's code response

#     Return:
#         blank_lines: int, number of blank lines
#     """
#     code_lines = ind_q_resp.splitlines()
#     blank_lines = 0

#     for line in code_lines:
#         if line.strip() == "":
#             blank_lines += 1

#     return blank_lines


# def blank_line_ratio(ind_q_resp: str) -> float:
#     """Feature 243: Returns blank lines divided by total lines.

#     Parameters:
#         ind_q_resp: str, one question's code response

#     Return:
#         blank_ratio: float, blank-line ratio
#     """
#     total_lines = total_line_count(ind_q_resp)
#     blank_lines = blank_line_count(ind_q_resp)

#     if total_lines == 0:
#         return 0.0

#     blank_ratio = blank_lines / total_lines

#     return blank_ratio


# def pure_comment_line_count(ind_q_resp: str) -> int:
#     """Returns the number of pure-comment lines.

#     A pure-comment line contains only optional whitespace followed by
#     a # comment.

#     Lines containing both code and an inline comment are classified as
#     code lines and are not counted.

#     Docstrings are not counted as comments.

#     Parameters:
#         ind_q_resp: str, one question's code response

#     Return:
#         pure_comment_lines: int, number of pure-comment lines
#     """
#     pure_comment_lines = 0

#     for line in ind_q_resp.splitlines():
#         if line.lstrip().startswith("#"):
#             pure_comment_lines += 1

#     return pure_comment_lines


# def comment_line_ratio(ind_q_resp: str) -> float:
#     """Feature 244: Returns pure-comment lines divided by total lines.

#     A line containing both code and a comment is treated as a code line.
#     Docstrings are not counted as comments.

#     Parameters:
#         ind_q_resp: str, one question's code response

#     Return:
#         comment_ratio: float, pure-comment-line ratio
#     """
#     total_lines = total_line_count(ind_q_resp)
#     pure_comment_lines = pure_comment_line_count(ind_q_resp)

#     if total_lines == 0:
#         return 0.0

#     comment_ratio = pure_comment_lines / total_lines

#     return comment_ratio

# FOLDER PROCESSING
def get_user_number(response_path: Path) -> int:
    """Extracts the user number from a folder such as 'User 24'."""
    for folder_name in response_path.parts:
        match = re.fullmatch(
            r"User\s+(\d+)",
            folder_name,
            flags=re.IGNORECASE
        )
        if match is not None:
            return int(match.group(1))
    raise ValueError(
        f"Could not determine user number from {response_path}"
    )

def write_prelim_dataset(
    dataset_folder: str,
    output_filename: str = "prelim_analysis_data.csv"
):
    """Processes every response file and writes an ANOVA-ready CSV."""

    session_conditions = {
        1: "bona_fide",
        2: "transcribe",
        3: "mimicry",
        4: "paraphrase",
        5: "deception"
    }

    dataset_path = Path(dataset_folder)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_folder}"
        )

    response_files = []

    for filepath in dataset_path.rglob("*.json"):
        if "responses" in filepath.name.lower():
            response_files.append(filepath)

    rows = []
    processed_user_sessions = set()

    for response_path in response_files:
        user_number = get_user_number(response_path)

        (session, double_quote_ratio,identifier_length_avg, unique_identifier_ratio, session_unique_identifier_ratio, blank_ratio,comment_ratio) = prelim_analysis(str(response_path))

        user_session = (user_number, session)

        if user_session in processed_user_sessions:
            raise ValueError(
                f"More than one response file was found for "
                f"User {user_number}, Session {session}"
            )

        processed_user_sessions.add(user_session)

        row = {
            "user": user_number,
            "session": session,
            "condition": session_conditions[session],
            "double_quote_preference_ratio": double_quote_ratio,
            "avg_identifier_length": identifier_length_avg,
            "ratio_different_identifiers": unique_identifier_ratio,
            "session_ratio_different_identifiers": session_unique_identifier_ratio,
            "blank_line_ratio": blank_ratio,
            "comment_line_ratio": comment_ratio
        }

        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["user"],
            row["session"]
        )
    )

    field_names = [
        "user",
        "session",
        "condition",
        "double_quote_preference_ratio",
        "avg_identifier_length",
        "ratio_different_identifiers",
        "session_ratio_different_identifiers",
        "blank_line_ratio",
        "comment_line_ratio"
    ]

    with open(
        output_filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=field_names
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Response files found: {len(response_files)}")
    print(f"Rows written: {len(rows)}")
    print(f"Output file: {output_filename}")
    

#GENERAL ANALYSIS FUNCTIONS

#SUMMER ANALYSIS-
#using double_quote_preference_ratio, avg_identifier_length, ratio_different_identifers, blank_line_ratio, comment_line_ratio

def prelim_analysis(filepath: str):
    double_quote_ratio = 0
    identifier_length_avg = 0 
    unique_identifier_ratio = 0 
    blank_ratio = 0
    comment_ratio = 0
    quote_q_num = 0
    identifier_q_num = 0
    all_identifier_list = []
    
    
    resp_list, session = open_file(filepath)
    code_tuple = each_q_code(resp_list)
    question_number = len(code_tuple)

    for ind_q_resp in code_tuple:
        identifier_list, keyword_list, token_list = (extract_code_feature_lists(ind_q_resp))
        if identifier_list is not None:
            all_identifier_list.extend(identifier_list)
        
        #Double Quote Ratio
        q_quote_ratio = double_quote_preference_ratio(ind_q_resp)
        
        if q_quote_ratio is not None: 
            double_quote_ratio += q_quote_ratio
            quote_q_num += 1

        #Avg identifier length
        #Ratio of unique identifiers
        if identifier_list is not None:
            identifier_length_avg += avg_identifier_length(identifier_list) #Avg identifier length
            unique_identifier_ratio += ratio_different_identifiers(identifier_list)#Ratio of unique identifiers

            identifier_q_num += 1

        #Blank line ratio
        blank_ratio += blank_line_ratio(ind_q_resp)

        #Comment line ratio
        comment_ratio += comment_line_ratio(ind_q_resp)

    #Getting averages across 6 questions
    session_unique_identifier_ratio = ratio_different_identifiers(all_identifier_list)
    if quote_q_num != 0:
        double_quote_ratio = double_quote_ratio/quote_q_num
    else:
        double_quote_ratio = None
    if identifier_q_num != 0:
        identifier_length_avg /= identifier_q_num
        unique_identifier_ratio /= identifier_q_num
    else:
        identifier_length_avg = None
        unique_identifier_ratio = None
    blank_ratio = blank_ratio/question_number
    comment_ratio = comment_ratio/question_number

    return session, double_quote_ratio, identifier_length_avg, unique_identifier_ratio, session_unique_identifier_ratio, blank_ratio, comment_ratio 

        
    
##def main():
##    resp_list, session = open_file()
##    code_tuple = each_q_code(resp_list)
##    exp_tuple = each_q_exp(resp_list)
##
##    print("CODE FEATURES")
##
##    for ind_q_resp in code_tuple:
##        total_char = tot_char(ind_q_resp, "code")
##        alpha_ratio = ratio_alpha_char(ind_q_resp, "code")
##        upper_ratio = ratio_upper_char(ind_q_resp, "code")
##        digit_ratio = ratio_digit_char(ind_q_resp, "code")
##        tab_ratio = ratio_tab_char(ind_q_resp)
##        for ind_q_resp in code_tuple:
##            identifier_list, keyword_list, token_list = (
##                extract_code_feature_lists(ind_q_resp)
##            )
##
##            if token_list is None:
##                total_tokens = None
##            else:
##                total_tokens = len(token_list)
##
##            total_characters = tot_char(ind_q_resp, "code")
##
##            # Features 54-55
##            feature_54 = total_identifier_count(identifier_list)
##            feature_55 = total_keyword_count(keyword_list)
##
##            # Features 56-57
##            feature_56 = ratio_short_identifiers(
##                identifier_list,
##                total_tokens
##            )
##            feature_57 = ratio_short_keywords(
##                keyword_list,
##                total_tokens
##            )
##
##            # Features 58-59
##            feature_58 = ratio_identifier_characters(
##                identifier_list,
##                total_characters
##            )
##            feature_59 = ratio_keyword_characters(
##                keyword_list,
##                total_characters
##            )
##
##            # Features 60-61
##            feature_60 = avg_identifier_length(identifier_list)
##            feature_61 = avg_keyword_length(keyword_list)
##
##            # Features 62-63
##            feature_62 = avg_line_length_characters(ind_q_resp)
##            feature_63 = avg_line_length_tokens(
##                ind_q_resp,
##                total_tokens
##            )
##
##            # Features 64-65
##            feature_64 = ratio_different_identifiers(
##                identifier_list
##            )
##            feature_65 = ratio_different_keywords(
##                keyword_list
##            )
##
##            print(
##                feature_54,
##                feature_55,
##                feature_56,
##                feature_57,
##                feature_58,
##                feature_59,
##                feature_60,
##                feature_61,
##                feature_62,
##                feature_63,
##                feature_64,
##                feature_65
##            )
##
##        print(total_char, alpha_ratio, upper_ratio, digit_ratio, tab_ratio)
##
##    print("EXPLANATION FEATURES")
##
##    for ind_q_resp in exp_tuple:
##        total_char = tot_char(ind_q_resp, "explanation")
##        alpha_ratio = ratio_alpha_char(ind_q_resp, "explanation")
##        upper_ratio = ratio_upper_char(ind_q_resp, "explanation")
##        digit_ratio = ratio_digit_char(ind_q_resp, "explanation")
##
##        print(total_char, alpha_ratio, upper_ratio, digit_ratio)
              
if __name__ == "__main__":
    dataset_folder = input(
        "Enter the path to the Dataset folder: "
    )

    write_prelim_dataset(dataset_folder)
    
