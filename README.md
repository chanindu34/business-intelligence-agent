# Business Intelligence Agent

**Live demo:** https://business-intelligence-agent-7h3fc6ewvtfnfg6oamjusb.streamlit.app

A tool-calling agent that decides, per question, whether to search John Keells Holdings' Annual Report 2025/26 (via retrieval-augmented generation), run a safe arithmetic calculation, or answer directly, without being told in advance which is needed.

## What it does

Ask a question. The agent looks at it and picks the right tool itself:

- **`search_knowledge_base`**: retrieval over the same 612-page annual report used in the standalone RAG Business Assistant project, wired in as a callable tool rather than a separate pipeline
- **`calculate`**: a safe, AST-based arithmetic evaluator (no `eval()`), supporting the standard operators
- **No tool**: the agent answers directly when neither is needed, or declines rather than guessing when a question falls outside both tools (see Known limitation below)

## Architecture

```
question → agent (Gemini function calling) → picks a tool or answers directly
                                                    |
                             +----------------------+----------------------+
                             |                                             |
                    calculate (AST eval)                    search_knowledge_base (RAG)
                             |                                             |
                        Pydantic-validated                        ChromaDB + Gemini embeddings
```

RAG is not a separate system the agent occasionally defers to. It is one of the agent's tools, exactly like the calculator.

## Testing results

5/5 test questions correctly routed and answered, including one deliberately ambiguous case ("how many risks does the report mention?") and one deliberate failure case (an unsupported math operation), both handled gracefully rather than crashing or hallucinating an answer.

## Known limitation, honestly documented

The agent behaves conservatively on general-knowledge questions outside its two tools (e.g. "what's the capital of France?"). It declines rather than answering from its own training knowledge. This is arguably a safer default for a business tool, but differs from general-purpose assistant behavior.

## Tech stack

Python, Google Gemini (generation + embeddings), ChromaDB, Pydantic, FastAPI, Streamlit, Docker, GitHub Actions (CI)

## Tests

```
pytest -v
```

Covers the calculator (multiplication, division, division by zero, invalid syntax) and Pydantic input validation for both tools. Runs automatically on every push and pull request via GitHub Actions.

## Setup

```
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
```

## Run it locally

FastAPI service:
```
uvicorn app:app --reload
```

Streamlit chat UI:
```
streamlit run streamlit_app.py
```

Or with Docker:
```
docker build -t bi-agent .
docker run -p 8000:8000 --env-file .env bi-agent
```

## Files

- `app.py`: agent loop, tool schemas, calculator, RAG retrieval, FastAPI app
- `streamlit_app.py`: chat UI over the same agent
- `test_pipeline.py`: pytest suite for the calculator and input validation
- `DESIGN.md`: architecture notes and testing results
- `Dockerfile`: containerized FastAPI service
- `.github/workflows/test.yml`: CI, runs the test suite on every push and PR
- `requirements.txt`: dependencies
- `chroma_db/`: persistent vector store, committed so the app runs out of the box without re-ingesting the report

## Author

Chanindu Dahanayake
