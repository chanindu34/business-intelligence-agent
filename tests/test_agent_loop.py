"""The agent loop against a scripted fake Gemini client, using real google.genai types."""
from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import errors, types

import agent
from agent import run_agent
from retrieval import ReportIndex

DOCS = ["Group recurring EBITDA rose 71% to Rs.78.05 billion",
        "Retail recurring EBITDA of Rs.31.74 billion, up 190%"]
INDEX = ReportIndex(["c0", "c1"], DOCS, np.eye(2).tolist(), {"c0": 14, "c1": 15})


def text(t):
    return types.Part(text=t)


def call(name, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def reply(*parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=types.Content(role="model", parts=list(parts)),
                                                       finish_reason="STOP")])


class ScriptedClient:
    """Returns the scripted replies in order; records what each request contained."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)


def run(script, question="q", **kw):
    client = ScriptedClient(script)
    return run_agent(question, client, index=INDEX, models=kw.pop("models", ["m1", "m2"]), **kw), client


def test_direct_answer_uses_no_tools():
    result, client = run([reply(text("I answer questions about the JKH annual report."))])
    assert result.tools_used == [] and result.turns == 1 and result.stopped == "answered"


def test_single_tool_then_answer():
    result, client = run([reply(call("calculate", expression="128/4")), reply(text("128 / 4 = 32"))])
    assert result.tools_used == ["calculate"] and result.steps[0]["result"]["result"] == "32"
    # The tool result went back to the model as a function_response.
    sent = client.requests[1]["contents"][-1].parts[0].function_response
    assert sent.name == "calculate" and sent.response["result"] == "32"


def test_search_then_calculate_chain():
    result, _ = run([
        reply(call("search_knowledge_base", query="recurring EBITDA Retail and Group")),
        reply(call("calculate", expression="31.74 / 78.05 * 100")),
        reply(text("Retail was 40.7% of Group recurring EBITDA (page 14, page 15).")),
    ])
    assert result.tools_used == ["search_knowledge_base", "calculate"]
    assert result.pages == [14, 15] and result.turns == 3


def test_parallel_tool_calls_in_one_reply_all_run():
    result, client = run([
        reply(call("calculate", expression="1+1"), call("calculate", expression="2+2")),
        reply(text("2 and 4")),
    ])
    assert [s["result"]["result"] for s in result.steps] == ["2", "4"]
    assert len(client.requests[1]["contents"][-1].parts) == 2  # both results sent back together


def test_text_before_a_tool_call_is_not_mistaken_for_the_answer():
    # The first version read only parts[0] and returned "Let me check." as the answer.
    result, _ = run([reply(text("Let me check."), call("calculate", expression="6*7")), reply(text("42"))])
    assert result.answer == "42" and result.tools_used == ["calculate"]


def test_thought_parts_are_not_shown_as_answer():
    result, _ = run([reply(types.Part(text="internal reasoning", thought=True), text("Final."))])
    assert result.answer == "Final."


def test_too_many_calls_in_one_turn_are_capped():
    calls = [call("calculate", expression=f"{i}+1") for i in range(6)]
    result, client = run([reply(*calls), reply(text("done"))])
    assert len(result.steps) == agent.MAX_CALLS_PER_TURN
    assert len(client.requests[1]["contents"][-1].parts) == 6  # skipped ones still get a response


def test_step_limit_stops_a_looping_model():
    script = [reply(call("calculate", expression="1+1")) for _ in range(agent.MAX_TURNS)]
    result, _ = run(script)
    assert result.stopped == "max_turns" and len(result.steps) == agent.MAX_TURNS


def test_bad_tool_arguments_come_back_as_errors_not_crashes():
    result, _ = run([reply(call("calculate")), reply(text("I need an expression."))])
    assert result.steps[0]["result"]["success"] is False and "invalid arguments" in result.steps[0]["result"]["error"]


def test_unknown_tool_is_reported_to_the_model():
    result, _ = run([reply(call("delete_everything")), reply(text("I can't do that."))])
    assert result.steps[0]["result"]["error"] == "unknown tool 'delete_everything'"


DAILY = {"error": {"code": 429, "message": "quota", "details": [{"quotaId": "GenerateRequestsPerDay"}]}}


def test_daily_quota_fails_over_to_next_model_and_restarts_cleanly():
    result, client = run([errors.ClientError(429, DAILY), reply(text("answer from m2"))])
    assert result.model == "m2" and [r["model"] for r in client.requests] == ["m1", "m2"]


def test_model_failing_mid_run_restarts_on_next_model():
    result, client = run([
        reply(call("calculate", expression="1+1")),
        errors.ServerError(503, {"error": {"code": 503, "message": "x"}}),
        errors.ServerError(503, {"error": {"code": 503, "message": "x"}}),
        reply(text("fresh answer from m2")),
    ])
    assert result.model == "m2" and result.steps == []  # m2 started from scratch


def test_overloaded_model_gets_one_quick_retry():
    result, client = run([errors.ServerError(503, {"error": {"code": 503, "message": "x"}}), reply(text("ok"))])
    assert result.model == "m1" and len(client.requests) == 2


def test_bad_request_is_not_retried():
    with pytest.raises(errors.ClientError):
        run([errors.ClientError(400, {"error": {"code": 400, "message": "bad"}})])


def test_no_model_available():
    result, _ = run([errors.ClientError(429, DAILY), errors.ClientError(404, {"error": {"code": 404, "message": "x"}})])
    assert result.stopped == "no_model" and "No model is available" in result.answer


def test_empty_reply_is_handled():
    empty = SimpleNamespace(candidates=[SimpleNamespace(content=None, finish_reason="SAFETY")])
    result, _ = run([empty])
    assert result.stopped == "blocked" and "SAFETY" in result.answer


def test_config_disables_sdk_auto_calling_and_sets_system_prompt():
    _, client = run([reply(text("hi"))])
    cfg = client.requests[0]["config"]
    assert cfg.automatic_function_calling.disable is True
    assert "ONLY source for facts" in cfg.system_instruction


def test_history_is_sent_for_follow_ups():
    history = [{"role": "user", "content": "How much did EBITDA grow?"}, {"role": "assistant", "content": "75%"}]
    _, client = run([reply(text("ok"))], question="Why?", history=history)
    roles = [c.role for c in client.requests[0]["contents"]]
    assert roles == ["user", "model", "user"]


def test_prompt_forbids_mental_maths_after_calculator_error_and_latex():
    from agent import SYSTEM_PROMPT
    assert "Never work out the result yourself" in SYSTEM_PROMPT
    assert "no LaTeX" in SYSTEM_PROMPT


def test_quota_exhausted_model_is_skipped_on_the_next_question():
    run([errors.ClientError(429, DAILY), reply(text("first"))])
    result, client = run([reply(text("second"))])
    assert [r["model"] for r in client.requests] == ["m2"]  # m1 not retried for an hour


def test_overloaded_model_cools_down_for_a_minute_only(monkeypatch):
    overload = errors.ServerError(503, {"error": {"code": 503, "message": "x"}})
    run([overload, overload, reply(text("first"))])
    assert agent._benched["m1"] - agent.time.monotonic() <= agent.OVERLOAD_COOLDOWN_SECONDS


def test_all_models_benched_still_tries_them():
    agent._benched.update({"m1": agent.time.monotonic() + 999, "m2": agent.time.monotonic() + 999})
    result, client = run([reply(text("ok"))])
    assert result.answer == "ok" and client.requests[0]["model"] == "m1"


def test_mental_arithmetic_is_sent_back_to_use_the_calculator():
    result, client = run([
        reply(text("Retail share: 31,740 / 78,048 x 100 = 40.67%")),
        reply(call("calculate", expression="31740/78048*100")),
        reply(text("Retail share: 31,740 / 78,048 x 100 = 40.67%")),
    ])
    assert result.tools_used == ["calculate"] and result.turns == 3
    assert "did not call the calculate tool" in client.requests[1]["contents"][-1].parts[0].text


def test_verifier_nudges_only_once():
    result, _ = run([reply(text("2 + 2 = 4")), reply(text("2 + 2 = 4"))])
    assert result.answer == "2 + 2 = 4" and result.turns == 2


def test_answer_after_calculator_is_not_sent_back():
    result, client = run([reply(call("calculate", expression="2+2")), reply(text("2 + 2 = 4"))])
    assert result.turns == 2 and len(client.requests) == 2
