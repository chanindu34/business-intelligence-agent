from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from google.genai import types

import app as app_module
from retrieval import ReportIndex

INDEX = ReportIndex(["c0"], ["Group EBITDA rose 75% to Rs.80.01 billion"], [[1.0]], {"c0": 14})


def reply(*parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=types.Content(role="model", parts=list(parts)))])


@pytest.fixture
def client(monkeypatch):
    script = [
        reply(types.Part(function_call=types.FunctionCall(name="search_knowledge_base", args={"query": "EBITDA"}))),
        reply(types.Part(text="Group EBITDA rose 75% to Rs.80.01 billion (page 14).")),
    ]
    fake = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: script.pop(0)))
    monkeypatch.setattr(app_module, "get_client", lambda: fake)
    monkeypatch.setattr(app_module, "get_index", lambda: INDEX)
    monkeypatch.setattr(app_module, "get_embedder", lambda: None)
    return TestClient(app_module.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_tools_and_pages(client):
    body = client.post("/ask", json={"question": "How much did EBITDA grow?"}).json()
    assert body["tools_used"] == ["search_knowledge_base"] and body["pages"] == [14]
    assert "80.01" in body["answer"] and body["stopped"] == "answered"


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x" * 501}, {}])
def test_bad_requests_rejected_before_any_model_call(client, payload):
    assert client.post("/ask", json=payload).status_code == 422
