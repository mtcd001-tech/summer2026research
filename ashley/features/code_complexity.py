import ast


class CyclomaticComplexityVisitor(ast.NodeVisitor):
    """Count decision points within a function."""

    def __init__(self):
        self.decision_count = 0

    def visit_If(self, node):
        self.decision_count += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.decision_count += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.decision_count += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.decision_count += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.decision_count += 1
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        """Do not count decisions inside nested functions."""
        return

    def visit_AsyncFunctionDef(self, node):
        """Do not count decisions inside nested async functions."""
        return

def function_cyclomatic_complexity(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef
) -> int:
    """Return the cyclomatic complexity of one function."""

    visitor = CyclomaticComplexityVisitor()

    for statement in function_node.body:
        visitor.visit(statement)

    return visitor.decision_count + 1

def average_cyclomatic_complexity(code: str) -> float | None:
    """Return the average cyclomatic complexity of functions in code.

    Complexity is calculated separately for each function and then
    averaged across all functions.

    Returns None if the code cannot be parsed or contains no functions.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    if len(functions) == 0:
        return None

    complexities = [
        function_cyclomatic_complexity(function)
        for function in functions
    ]

    return sum(complexities) / len(complexities)

class NestingDepthVisitor(ast.NodeVisitor):
    """Find the maximum nesting depth of control structures."""

    def __init__(self):
        self.current_depth = 0
        self.max_depth = 0

    def visit_control_structure(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)

        self.generic_visit(node)

        self.current_depth -= 1

    def visit_If(self, node):
        self.visit_control_structure(node)

    def visit_For(self, node):
        self.visit_control_structure(node)

    def visit_AsyncFor(self, node):
        self.visit_control_structure(node)

    def visit_While(self, node):
        self.visit_control_structure(node)

    def visit_Try(self, node):
        self.visit_control_structure(node)

    def visit_TryStar(self, node):
        self.visit_control_structure(node)

    def visit_FunctionDef(self, node):
        previous_depth = self.current_depth
        self.current_depth = 0

        for statement in node.body:
            self.visit(statement)

        self.current_depth = previous_depth

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.visit_FunctionDef(node)

def maximum_nesting_depth(code: str) -> int | None:
    """Return the maximum nesting depth of control structures in code.

    Function and class definitions do not count toward nesting depth.
    Returns None if the code cannot be parsed.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    visitor = NestingDepthVisitor()
    visitor.visit(tree)

    return visitor.max_depth