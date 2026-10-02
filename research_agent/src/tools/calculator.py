import ast
import operator
import math
from langchain_core.tools import tool

# Supported operators mapping
SUPPORTED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SUPPORTED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise TypeError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.Num):  # For older Python compatibility
        return node.n
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in SUPPORTED_OPERATORS:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return SUPPORTED_OPERATORS[op_type](left, right)
        raise TypeError(f"Unsupported operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in SUPPORTED_OPERATORS:
            operand = _eval_node(node.operand)
            return SUPPORTED_OPERATORS[op_type](operand)
        raise TypeError(f"Unsupported unary operator: {op_type}")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SUPPORTED_FUNCTIONS:
            func = SUPPORTED_FUNCTIONS[node.func.id]
            if callable(func):
                args = [_eval_node(arg) for arg in node.args]
                return func(*args)
        raise TypeError(f"Unsupported function call")
    elif isinstance(node, ast.Name):
        if node.id in SUPPORTED_FUNCTIONS:
            val = SUPPORTED_FUNCTIONS[node.id]
            if not callable(val):
                return val
        raise TypeError(f"Unsupported variable/constant: {node.id}")
    else:
        raise TypeError(f"Unsupported syntax: {type(node)}")


@tool
def safe_calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely using AST parsing.

    Args:
        expression: A math expression string (e.g. '2 + 2 * (3 / 4)' or 'sqrt(16)').
    """
    try:
        # Clean expression
        expression = expression.strip()
        tree = ast.parse(expression, mode='eval')
        result = _eval_node(tree.body)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: Could not evaluate expression '{expression}'. Error: {str(e)}"
