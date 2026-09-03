# Business Intelligence Agent — Capstone Architecture

## Problem
Business users need to ask natural-language questions that sometimes require
calculation, sometimes require searching company documents, and sometimes
neither — without knowing in advance which is needed.

## Architecture: Agent-First
The agent (Week 3) is the entry point. It decides, per question, whether to:
- Call `calculate` (safe AST-based evaluator)
- Call `search_knowledge_base` (Week 2's RAG retrieval, wired in as a tool)
- Answer directly (no tool needed)

RAG is not a separate system the agent occasionally defers to — it IS one of
the agent's tools, exactly like the calculator. This was true since Day 17;
Week 4 formalizes it with production infrastructure (API, tests, deployment).

## Tech stack
Python, Google Gemini (generation + embeddings), ChromaDB, Pydantic, FastAPI,
Docker, GitHub Actions.

## Success metrics (real, from Day 21 testing)
5/5 test questions correctly routed and answered, including one deliberately
ambiguous case ("how many risks does the report mention?") and one deliberate
failure case (unsupported math operation), both handled gracefully.

## Known limitation, honestly documented
The agent behaves conservatively on general-knowledge questions outside its
two tools (e.g. "what's the capital of France?") — it declines rather than
answering from its own training knowledge. This is arguably a safer default
for a business tool, but differs from general-purpose assistant behavior.