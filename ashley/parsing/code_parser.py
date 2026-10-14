import ast
import io
import keyword
import tokenize

class IdentifierCollector(ast.NodeVisitor):
    """Find names that are defined by the programmer.

    Includes variables, function names, classes, function parameters,
    imported local names, exception variables, attributes, and methods.
    """

    def __init__(self):
        self.user_defined_names = set()
        self.user_defined_attributes = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.user_defined_names.add(node.id)

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.user_defined_names.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.user_defined_names.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.user_defined_names.add(node.name)

        for statement in node.body:
            if isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                self.user_defined_attributes.add(statement.name)

        self.generic_visit(node)

    def visit_arg(self, node):
        self.user_defined_names.add(node.arg)
        self.generic_visit(node)

    def visit_alias(self, node):
        if node.asname is not None:
            imported_name = node.asname
        else:
            imported_name = node.name.split(".")[0]

        if imported_name != "*":
            self.user_defined_names.add(imported_name)

        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name is not None:
            self.user_defined_names.add(node.name)

        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Store):
            self.user_defined_attributes.add(node.attr)

        self.generic_visit(node)

def get_imported_name(node: ast.alias) -> str:
    """Return the local name introduced by an import."""

    if node.asname is not None:
        return node.asname

    return node.name.split(".")[0]


def extract_identifier_list(code: str) -> list | None:
    """Return every occurrence of a user-defined identifier.

    Repeated occurrences are preserved.

    Returns None when the code cannot be parsed as valid Python.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    collector = IdentifierCollector()
    collector.visit(tree)

    defined_names = collector.user_defined_names
    defined_attributes = collector.user_defined_attributes

    identifiers = []

    for node in ast.walk(tree):

        if isinstance(node, ast.Name):
            if node.id in defined_names:
                identifiers.append(node.id)

        elif isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if node.name in defined_names:
                identifiers.append(node.name)

        elif isinstance(node, ast.ClassDef):
            if node.name in defined_names:
                identifiers.append(node.name)

        elif isinstance(node, ast.arg):
            if node.arg in defined_names:
                identifiers.append(node.arg)

        elif isinstance(node, ast.alias):
            imported_name = get_imported_name(node)

            if imported_name in defined_names:
                identifiers.append(imported_name)

        elif isinstance(node, ast.ExceptHandler):
            if (
                node.name is not None
                and node.name in defined_names
            ):
                identifiers.append(node.name)

        elif isinstance(node, ast.Attribute):
            if node.attr in defined_attributes:
                identifiers.append(node.attr)

    return identifiers

def extract_keyword_list(code: str) -> list | None:
    """Return every occurrence of a Python keyword.

    Keywords inside comments and strings are not counted.
    """

    keywords = []

    try:
        tokens = tokenize.generate_tokens(
            io.StringIO(code).readline
        )

        for token in tokens:
            if (
                token.type == tokenize.NAME
                and keyword.iskeyword(token.string)
            ):
                keywords.append(token.string)

    except (tokenize.TokenError, IndentationError):
        return None

    return keywords

def extract_code_token_list(code: str) -> list | None:
    """Return meaningful lexical tokens from a code response."""

    ignored_token_types = {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }

    token_list = []

    try:
        tokens = tokenize.generate_tokens(
            io.StringIO(code).readline
        )

        for token in tokens:
            if (
                token.type not in ignored_token_types
                and token.string.strip() != ""
            ):
                token_list.append(token.string)

    except (tokenize.TokenError, IndentationError):
        return None

    return token_list

def extract_code_feature_lists(code: str) -> tuple:
    """Return reusable identifier, keyword, and token lists."""

    identifiers = extract_identifier_list(code)
    keywords = extract_keyword_list(code)
    tokens = extract_code_token_list(code)

    return identifiers, keywords, tokens


#TEST
if __name__ == "__main__":
    test_code = """
def find_highest(scores):
    highest = scores[0]
    for score in scores:
        if score > highest:
            highest = score

    return highest
    """

    print("CODE BEING PARSED:")
    print(repr(test_code))

    identifiers, keywords, tokens = extract_code_feature_lists(test_code)

    print("\nIdentifiers:")
    print(identifiers)

    print("\nKeywords:")
    print(keywords)

    print("\nTokens:")
    print(tokens)