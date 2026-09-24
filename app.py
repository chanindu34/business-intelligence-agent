"""
Business Intelligence Agent

A tool-calling agent that decides, per question, whether to search a
company annual report (RAG), run a safe arithmetic calculation, or
answer directly. Exposes both a FastAPI service and (via streamlit_app.py)
a Streamlit chat UI.
"""

import ast
import logging
import operator
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from chromadb import EmbeddingFunction
from fastapi import FastAPI
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
COLLECTION_NAME = "day10_documents"
CHROMA_DB_PATH = "chroma_db"

MAX_GENERATION_RETRIES = 5
MAX_EMBEDDING_RETRIES = 5
GENERATION_RETRY_BASE_SECONDS = 15
MAX_AGENT_TURNS = 5
DEFAULT_TOP_K = 6

client_genai = genai.Client()


def generate_with_retry(contents: List, config: Any, max_retries: int = MAX_GENERATION_RETRIES):
    """Call the generation API, retrying with linear backoff on failure."""
    for attempt in range(max_retries):
        try:
            return client_genai.models.generate_content(
                model=GENERATION_MODEL,
                contents=contents,
                config=config
            )
        except Exception:
            wait = GENERATION_RETRY_BASE_SECONDS * (attempt + 1)
            logger.warning(
                "Generation attempt %d/%d failed, retrying in %ds",
                attempt + 1, max_retries, wait, exc_info=True
            )
            time.sleep(wait)
    raise RuntimeError("Failed to generate after retries.")


# ==========================================
# Safe AST-based calculator
# ==========================================
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        left_val = _eval_node(node.left)
        right_val = _eval_node(node.right)
        op_type = type(node.op)
        if op_type in OPERATORS:
            return OPERATORS[op_type](left_val, right_val)
        raise ValueError(f"Operator {op_type.__name__} not supported.")
    elif isinstance(node, ast.UnaryOp):
        operand_val = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type in OPERATORS:
            return OPERATORS[op_type](operand_val)
        raise ValueError(f"Operator {op_type.__name__} not supported.")
    else:
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression via AST parsing (no eval())."""
    try:
        parsed_ast = ast.parse(expression.strip(), mode='eval')
        result = _eval_node(parsed_ast.body)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: Invalid expression ({str(e)})."


# ==========================================
# Embedding + retrieval
# ==========================================
def get_embedding(text: str, max_retries: int = MAX_EMBEDDING_RETRIES) -> List[float]:
    """Embed text via the Gemini embedding API, retrying with exponential backoff."""
    for attempt in range(max_retries):
        try:
            result = client_genai.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text
            )
            return result.embeddings[0].values
        except Exception:
            logger.warning("Embedding attempt %d/%d failed", attempt + 1, max_retries, exc_info=True)
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to embed after retries.")


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding function backed by the Gemini embedding API."""

    def __call__(self, input: List[str]) -> List[List[float]]:
        return [get_embedding(text) for text in input]


db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = db_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=GeminiEmbeddingFunction()
)


def retrieve(query: str, k: int = DEFAULT_TOP_K) -> List[str]:
    """Retrieve the top-k most relevant chunks for a query."""
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]


def search_knowledge_base(query: str) -> str:
    """Tool entry point: search the annual report and return joined chunks."""
    chunks = retrieve(query)
    return "\n\n".join(chunks)


# ==========================================
# Tool schemas
# ==========================================
calculate_tool_schema = {
    "name": "calculate",
    "description": "Evaluates a mathematical expression and returns the numeric result. Use this whenever the user asks a math question or needs a calculation.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "A mathematical expression, e.g. '47 * 12'"}
        },
        "required": ["expression"]
    }
}

search_kb_tool_schema = {
    "name": "search_knowledge_base",
    "description": "Searches the company's annual report for relevant information. Use this whenever the user asks a question about business performance, risks, strategy, financials, or anything that would be found in a corporate report. Not for math calculations.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query, phrased as a natural-language question about the report's content."}
        },
        "required": ["query"]
    }
}

both_tools = types.Tool(function_declarations=[calculate_tool_schema, search_kb_tool_schema])
config = types.GenerateContentConfig(tools=[both_tools])


# ==========================================
# Pydantic-validated tool execution
# ==========================================
class CalculateInput(BaseModel):
    expression: str


class SearchKBInput(BaseModel):
    query: str


def run_tool(tool_name: str, tool_args: Dict) -> str:
    """Validate tool arguments with Pydantic and dispatch to the right tool."""
    try:
        if tool_name == "search_knowledge_base":
            validated = SearchKBInput(**tool_args)
            return search_knowledge_base(validated.query)
        elif tool_name == "calculate":
            validated = CalculateInput(**tool_args)
            return calculate(validated.expression)
        else:
            return f"Error: unknown tool '{tool_name}' requested."
    except ValidationError as e:
        return f"Error: invalid arguments for '{tool_name}' ({e})."
    except Exception as e:
        logger.error("Tool execution failed for '%s'", tool_name, exc_info=True)
        return f"Error: tool execution failed ({str(e)})."


# ==========================================
# The agent loop
# ==========================================
def run_agent(question: str, max_turns: int = MAX_AGENT_TURNS) -> str:
    """Run the agent loop: let the model call tools until it produces a final answer."""
    conversation = [types.Content(role="user", parts=[types.Part(text=question)])]

    for turn in range(max_turns):
        response = generate_with_retry(conversation, config)
        part = response.candidates[0].content.parts[0]

        if not part.function_call:
            return response.text

        conversation.append(types.Content(role="model", parts=[part]))

        tool_name = part.function_call.name
        tool_args = part.function_call.args

        logger.info("Turn %d: calling tool %s with %r", turn + 1, tool_name, tool_args)

        result = run_tool(tool_name, tool_args)

        conversation.append(types.Content(role="user", parts=[
            types.Part.from_function_response(name=tool_name, response={"result": result})
        ]))

    return "Reached maximum tool-call turns without a final answer."


# ==========================================
# FastAPI wrapper
# ==========================================
app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QuestionRequest):
    answer = run_agent(request.question)
    return {"answer": answer}
