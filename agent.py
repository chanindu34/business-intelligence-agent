"""
Tool-calling agent: Gemini decides, per question, whether to search the
annual report, calculate, do both in sequence, or answer directly.

Loop
    1. Send the conversation + tool schemas to the model.
    2. Read EVERY part of the reply. Text parts are the answer; function_call
       parts are tool requests (there can be several in one reply).
    3. Run each requested tool, send all results back in one message.
    4. Repeat until the model answers without calling a tool, or a limit hits.

Safety limits: max turns, max tool calls per turn, a wall-clock budget, and
validated tool arguments. Tools have no side effects, so if a model fails
mid-run the whole run restarts cleanly on the next model in the chain.
"""

import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from google.genai import errors, types

from tools import TOOL_SCHEMAS, execute_tool

logger = logging.getLogger(__name__)

MODELS = [m.strip() for m in os.environ.get(
    "BI_MODELS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash,gemini-2.5-flash"
).split(",") if m.strip()]
EMBEDDING_MODEL = "gemini-embedding-001"
MAX_TURNS = 6
MAX_CALLS_PER_TURN = 4
TIME_BUDGET_SECONDS = 90

SYSTEM_PROMPT = """You are a business intelligence assistant for John Keells Holdings' (JKH) Annual Report 2025/26.

Tools:
- search_knowledge_base: the ONLY source for facts about JKH. Never state a company fact, figure or name from memory.
- calculate: use for ALL arithmetic (percentages, growth, differences, ratios). Never do arithmetic in your head.

Rules:
1. For any question about JKH, search first. Search again with different words if the first results miss.
2. If a calculation needs report figures, search for them first, then call calculate with those exact numbers.
3. Cite report facts with their page, for example (page 13). Show the calculation you ran.
4. If the passages do not contain the answer, say the report excerpts do not cover it. Do not guess.
5. For questions unrelated to JKH and not arithmetic, say briefly that you answer questions about the JKH 2025/26 annual report and calculations.
6. If calculate returns an error, tell the user it cannot be computed and why. Never work out the result yourself and never retry the same calculation in another form.
7. Answer only what was asked. Use the latest financial year (2025/26) unless the user asks for another year or a comparison.
8. Plain text only: no LaTeX, no $ signs for maths. Write calculations like: 31,740 / 78,048 x 100 = 40.67%.
9. Be concise: a direct answer first, then the figures with pages, then the calculation."""

TOOLS = types.Tool(function_declarations=TOOL_SCHEMAS)

# "31,740 / 78,048 x 100 = 40.67%" style working: a number, an operator, a number, then "= number".
_ARITHMETIC = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:/|x|×|\*|÷|\+|-)\s*\d[\d,]*(?:\.\d+)?[^=\n]{0,40}=\s*-?[\d,]*\.?\d")
CALCULATOR_NUDGE = ("Your answer shows a calculation, but you did not call the calculate tool. "
                    "Call calculate with that expression now, then give the final answer using its result.")


def _shows_unchecked_arithmetic(text: str) -> bool:
    return bool(_ARITHMETIC.search(text or ""))


class ModelUnavailable(Exception):
    """This model cannot serve the request (quota, missing, overloaded)."""


@dataclass
class AgentResult:
    answer: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    model: Optional[str] = None
    turns: int = 0
    stopped: str = "answered"  # answered | max_turns | time_budget | no_model | blocked
    seconds: float = 0.0

    @property
    def tools_used(self) -> List[str]:
        return [s["tool"] for s in self.steps]

    @property
    def pages(self) -> List[int]:
        return sorted({p["page"] for s in self.steps for p in s.get("result", {}).get("passages", []) if p.get("page")})


def _is_daily_quota(e: Exception) -> bool:
    return "PerDay" in str(e) or "per day" in str(e).lower()


def _call_model(client, model: str, contents: list, config) -> Any:
    """One model call with fast failover rules:
    daily quota / missing model -> ModelUnavailable at once; 503 -> one quick retry;
    per-minute 429 -> one short wait; other 4xx -> raise (retrying cannot help)."""
    for attempt in range(2):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.ClientError as e:
            if e.code == 404 or (e.code == 429 and _is_daily_quota(e)):
                raise ModelUnavailable(f"{model}: {e.code}") from e
            if e.code == 429 and attempt == 0:
                m = re.search(r"retry in ([\d.]+)s", str(e))
                wait = float(m.group(1)) if m else 2.0
                if wait <= 10:
                    time.sleep(wait + random.uniform(0, 0.5))
                    continue
                raise ModelUnavailable(f"{model}: rate limited for {wait:.0f}s") from e
            if e.code == 429:
                raise ModelUnavailable(f"{model}: rate limited") from e
            raise
        except errors.ServerError as e:
            if attempt == 0:
                time.sleep(0.5 + random.uniform(0, 0.5))
                continue
            raise ModelUnavailable(f"{model}: server error {e.code}") from e
    raise ModelUnavailable(model)


def _history_contents(history: Optional[List[Dict[str, str]]]) -> list:
    """Previous turns as plain text, last 3 exchanges, so follow-ups have context."""
    out = []
    for m in (history or [])[-6:]:
        role = "user" if m["role"] == "user" else "model"
        out.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return out


def _run_with_model(client, model, question, history, index, embed, on_step, deadline) -> AgentResult:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[TOOLS],
        temperature=0.0,
        # We run tools ourselves (validation, limits, logging); never let the SDK auto-run them.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = _history_contents(history) + [types.Content(role="user", parts=[types.Part(text=question)])]
    steps: List[Dict[str, Any]] = []
    nudged = False

    for turn in range(1, MAX_TURNS + 1):
        if time.monotonic() > deadline:
            return AgentResult("I ran out of time before finishing. Please try a narrower question.",
                               steps, model, turn - 1, "time_budget")
        response = _call_model(client, model, contents, config)
        candidate = (response.candidates or [None])[0]
        content = getattr(candidate, "content", None)
        parts = list(getattr(content, "parts", None) or [])
        if not parts:
            reason = str(getattr(candidate, "finish_reason", "unknown"))
            return AgentResult(f"The model returned no answer (finish reason: {reason}).", steps, model, turn, "blocked")

        calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)).strip()

        if not calls:
            used_calculator = any(st["tool"] == "calculate" for st in steps)
            if (not nudged and not used_calculator and turn < MAX_TURNS
                    and _shows_unchecked_arithmetic(text)):
                # The answer shows a sum the calculator never ran. Send it back once.
                nudged = True
                contents.append(content)
                contents.append(types.Content(role="user", parts=[types.Part(text=CALCULATOR_NUDGE)]))
                if on_step:
                    on_step({"event": "verifier", "detail": "arithmetic without calculator; asked model to use it"})
                continue
            return AgentResult(text or "The model returned an empty answer.", steps, model, turn)

        # Send back the model's whole message (keeps any thought signatures newer
        # Gemini models require), then one message holding every tool result.
        contents.append(content)
        responses = []
        for fc in calls[:MAX_CALLS_PER_TURN]:
            args = dict(fc.args or {})
            if on_step:
                on_step({"event": "tool_start", "tool": fc.name, "args": args})
            t0 = time.monotonic()
            result = execute_tool(fc.name, args, index=index, embed=embed)
            step = {"turn": turn, "tool": fc.name, "args": args, "result": result,
                    "seconds": round(time.monotonic() - t0, 2)}
            steps.append(step)
            if on_step:
                on_step({"event": "tool_end", **step})
            responses.append(types.Part(function_response=types.FunctionResponse(
                id=getattr(fc, "id", None), name=fc.name, response=result)))
        for fc in calls[MAX_CALLS_PER_TURN:]:
            responses.append(types.Part(function_response=types.FunctionResponse(
                id=getattr(fc, "id", None), name=fc.name,
                response={"success": False, "error": f"skipped: at most {MAX_CALLS_PER_TURN} tool calls per turn"})))
        contents.append(types.Content(role="user", parts=responses))

    return AgentResult("I reached the step limit before finishing. Please try a narrower question.",
                       steps, model, MAX_TURNS, "max_turns")


def make_embedder(client) -> Callable[[str], list]:
    def embed(text: str) -> list:
        return client.models.embed_content(model=EMBEDDING_MODEL, contents=text).embeddings[0].values
    return embed


# Models that recently failed are skipped for a while so every question does not
# pay for the same failed call again. Daily quota or missing model: 1 hour. Overload: 1 minute.
QUOTA_COOLDOWN_SECONDS = 3600
OVERLOAD_COOLDOWN_SECONDS = 60
_benched: Dict[str, float] = {}


def _cooldown_for(reason: str) -> float:
    short = ("server error" in reason) or ("rate limited" in reason)
    return OVERLOAD_COOLDOWN_SECONDS if short else QUOTA_COOLDOWN_SECONDS


def _available(models: List[str]) -> List[str]:
    """Models not on cooldown, in order. If all are benched, try them all anyway."""
    now = time.monotonic()
    ready = [m for m in models if _benched.get(m, 0) <= now]
    return ready or list(models)


def run_agent(question: str, client, index=None, history=None, embed=None,
              on_step: Optional[Callable[[Dict], None]] = None, models: Optional[List[str]] = None) -> AgentResult:
    """Answer one question. Tries each model in order; a model that fails at any
    turn is abandoned and the run restarts on the next (tools have no side effects)."""
    start = time.monotonic()
    deadline = start + TIME_BUDGET_SECONDS
    failures = []
    for model in _available(list(models or MODELS)):
        try:
            result = _run_with_model(client, model, question, history, index, embed, on_step, deadline)
            result.seconds = round(time.monotonic() - start, 2)
            _benched.pop(model, None)
            if failures:
                logger.info("Answered with %s after: %s", model, "; ".join(failures))
            return result
        except ModelUnavailable as e:
            failures.append(str(e))
            _benched[model] = time.monotonic() + _cooldown_for(str(e))
            logger.warning("Model unavailable, trying next: %s", e)
            if on_step:
                on_step({"event": "model_failover", "detail": str(e)})
    return AgentResult("No model is available right now (" + "; ".join(failures) + "). Please try again later.",
                       [], None, 0, "no_model", round(time.monotonic() - start, 2))
