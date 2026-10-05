import evaluate


def test_alternative_tool_set_is_accepted():
    q = {"tools": ["search_knowledge_base"], "also_ok_tools": [["search_knowledge_base", "calculate"]]}
    assert evaluate.tools_match(["calculate", "search_knowledge_base"], q)
    assert evaluate.tools_match(["search_knowledge_base"], q)
    assert not evaluate.tools_match([], q)


def test_units_written_differently_still_match():
    assert evaluate.contains("grew by 74.50% (page 14)", "74.5")
    assert evaluate.contains("Rs. 80.01 billion", "80.01")


def test_questions_with_no_model_are_not_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    ok = {"id": "a", "expected_tools": [], "tools_used": [], "tools_ok": True, "answer_ok": True,
          "problems": [], "model": "m1", "turns": 1, "seconds": 1.0}
    skipped = {"id": "b", "expected_tools": [], "not_run": True, "model": None}
    text = evaluate.report([ok, skipped]).read_text()
    assert "1 questions scored; 1 not run" in text and "100% (1/1)" in text
    assert "Models that answered: m1 (1)" in text

