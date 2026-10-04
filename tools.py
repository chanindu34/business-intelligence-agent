"""
The agent's tools: a safe calculator and a search over the annual report.

Every tool returns a JSON-serialisable dict that always says whether it
succeeded, so the model can recover from a failed call instead of the
agent crashing.
"""

import ast
import logging
import math
import operator
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------
MAX_EXPRESSION_CHARS = 200
MAX_NODES = 60
MAX_EXPONENT = 100
MAX_POWER_BASE = 1e6
MAX_RESULT = 1e100

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


class CalculationError(ValueError):
    pass


def _eval(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            # 9**9**9 has ~370 million digits: without these limits one request
            # freezes the server. This was a real bug in the first version.
            if abs(right) > MAX_EXPONENT or abs(left) > MAX_POWER_BASE:
                raise CalculationError(f"exponent or base too large (limits: exponent {MAX_EXPONENT}, base {MAX_POWER_BASE:g})")
        result = _BINARY[type(node.op)](left, right)
        if isinstance(result, complex) or abs(result) > MAX_RESULT:
            raise CalculationError("result too large")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise CalculationError(f"unsupported syntax: {type(node).__name__}")


def _format(value) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise CalculationError("result is not a finite number")
        return f"{value:.10g}"
    return str(value)


def calculate(expression: str) -> Dict[str, Any]:
    """Safely evaluate arithmetic. Only numbers and + - * / // % ** and brackets."""
    expr = (expression or "").strip().replace(",", "").replace("^", "**")
    try:
        if not expr:
            raise CalculationError("empty expression")
        if len(expr) > MAX_EXPRESSION_CHARS:
            raise CalculationError(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
        tree = ast.parse(expr, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
            raise CalculationError("expression too complex")
        value = _eval(tree.body)
        return {"success": True, "expression": expr, "result": _format(value)}
    except ZeroDivisionError:
        return {"success": False, "expression": expr, "error": "division by zero"}
    except SyntaxError:
        return {"success": False, "expression": expr, "error": "invalid syntax; use numbers and + - * / ** ( )"}
    except (CalculationError, OverflowError) as e:
        return {"success": False, "expression": expr, "error": (str(e) or "result too large") + ". Do not compute this yourself; tell the user it is outside the calculator's limits."}


# ---------------------------------------------------------------------------
# Tool schemas (what the model sees) and validated inputs (what we accept)
# ---------------------------------------------------------------------------
CALCULATE_SCHEMA = {
    "name": "calculate",
    "description": (
        "Evaluate an arithmetic expression exactly. Use for ANY arithmetic, including "
        "percentages, growth rates, differences and ratios. Supports + - * / // % ** and brackets."
    ),
    "parameters": {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "For example '(80.01 - 45.85) / 45.85 * 100'"}},
        "required": ["expression"],
    },
}

SEARCH_SCHEMA = {
    "name": "search_knowledge_base",
    "description": (
        "Search John Keells Holdings' Annual Report 2025/26 (PDF pages 1-60 and 139-160: overview, "
        "management discussion, financial review, outlook and risks). The ONLY source for facts about "
        "the company. Returns passages with page numbers."
    ),
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to look up, in plain words"}},
        "required": ["query"],
    },
}

TOOL_SCHEMAS = [CALCULATE_SCHEMA, SEARCH_SCHEMA]


class CalculateInput(BaseModel):
    expression: str = Field(min_length=1, max_length=MAX_EXPRESSION_CHARS)


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)


def search_knowledge_base(query: str, index, embed: Optional[Callable[[str], list]] = None) -> Dict[str, Any]:
    """Hybrid search. If embedding fails, falls back to keyword search only."""
    vector, note = None, None
    if embed is not None:
        try:
            vector = embed(query)
        except Exception as e:
            note = f"semantic search unavailable ({type(e).__name__}); keyword search only"
            logger.warning(note)
    found = index.search(query, vector)
    passages = [
        {"source": f"S{n}", "page": p["page"], "text": p["text"]}
        for n, p in enumerate(found["passages"], 1)
    ]
    result = {"success": bool(passages), "method": found["method"], "passages": passages}
    if note:
        result["note"] = note
    if not passages:
        result["error"] = "no matching passages"
    return result


def execute_tool(name: str, args: Dict[str, Any], index=None, embed=None) -> Dict[str, Any]:
    """Validate arguments and run a tool. Never raises: errors come back as data."""
    try:
        if name == "calculate":
            return calculate(CalculateInput(**args).expression)
        if name == "search_knowledge_base":
            if index is None:
                return {"success": False, "error": "knowledge base not loaded"}
            return search_knowledge_base(SearchInput(**args).query, index, embed)
        return {"success": False, "error": f"unknown tool '{name}'"}
    except ValidationError as e:
        return {"success": False, "error": f"invalid arguments for {name}: {e.errors()[0]['msg']}"}
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return {"success": False, "error": f"{name} failed: {type(e).__name__}"}
