# Business Intelligence Agent

[![tests](https://github.com/chanindu34/business-intelligence-agent/actions/workflows/test.yml/badge.svg)](https://github.com/chanindu34/business-intelligence-agent/actions/workflows/test.yml)

**Live demo:** https://business-intelligence-agent-7h3fc6ewvtfnfg6oamjusb.streamlit.app

A tool-calling agent for John Keells Holdings' Annual Report 2025/26. For each question, Gemini decides whether to search the report, run a calculation, do both in sequence, or answer directly, and the UI shows every tool call as it happens.

Example: *"What share of the Group's recurring EBITDA came from Retail?"* The agent searches the report, finds Rs.31.74 billion (Retail) and Rs.78.05 billion (Group), calls the calculator with `31.74 / 78.05 * 100`, and answers 40.7% with page citations. It never does arithmetic in its head.

Companion project: [RAG Business Assistant](https://github.com/chanindu34/rag-business-assistant), which goes deeper on retrieval over the same report.

## How the agent works

```
question (+ last 3 exchanges)
   -> Gemini with a system prompt and two tool schemas
   -> reply parts:  text            -> final answer
                    function_call(s) -> run each tool, send all results back -> loop
   -> limits: 6 turns, 4 tool calls per turn, 90 s wall clock
```

| Tool | What it does | Guard rails |
|---|---|---|
| `search_knowledge_base` | Hybrid search: BM25 (stemmed) + Gemini embeddings, fused with Reciprocal Rank Fusion (k = 60), reranked by a cross-encoder, returns 5 passages with PDF page numbers | Embedding failure falls back to keyword search; reranker failure falls back to fused order |
| `calculate` | Exact arithmetic by walking the Python AST (no `eval`) | 200 chars, 60 AST nodes, exponent at most 100, results capped at 1e100, only numbers and operators |

**Grounding rules (system prompt):** company facts only from search, all arithmetic through the calculator, cite pages, say so when the passages do not cover it, decline unrelated general-knowledge questions.

**Model failover:** answers try `gemini-3.8-flash`, then `3.7-flash`, `3.5-flash`, `2.5-flash` (set `BI_MODELS` to change). A daily-quota or missing-model error moves to the next model at once; a 503 gets one quick retry; a bad request is never retried. Tools have no side effects, so a model that fails mid-run is abandoned and the run restarts cleanly on the next one.

**Index:** 820 chunks from PDF pages 1 to 60 and 139 to 160 (82 of 612 pages: overview, management discussion, financial review, outlook and risks). `data/chunk_pages.json` maps every chunk to its page; it was built by matching chunk text against the PDF, so citations needed no re-embedding.

## What testing found

1. **One message could freeze the server.** The "safe" calculator evaluated `9**9**9`, a number with about 370 million digits; it was still running when I killed it after 5 seconds. The calculator now rejects it in under 0.1 s, and a test enforces that.
2. **The agent read only the first part of each reply.** If Gemini wrote "Let me check." before a tool call, that sentence was returned as the answer and the tool never ran; a second tool call in the same reply was dropped. The loop now reads every part and runs every call (capped at 4 per turn).
3. **Retries could block for almost 4 minutes.** Every error, including ones retrying cannot fix, was retried 5 times with waits of 15 to 75 s. Failures now fall through to the next model in under a second.
4. **Nothing stopped answers from memory.** With no system prompt, the model could state company figures without searching. The grounding rules above now require a search for any company fact.
5. **The reranker was silently off** in an unreleased change: the model id `cross-encoder/mmarco-MiniLMv2-L12-H384-v1` does not exist (the real one is `mmarco-mMiniLMv2`), and the load error was swallowed. The reranker is now a model that exists, and a failure is logged and visible in the tool result.
6. **The tests had drifted from the code** (the calculator's return type changed, the tests did not). They are now 65 tests using a scripted fake Gemini client, so CI needs no API key.
7. **The prompt alone did not stop mental arithmetic.** On the live app, the model once showed `31,740 / 78,048 x 100 = 40.67%` without calling the calculator. A check in code now spots shown working that the calculator never ran and sends the answer back once with an instruction to use the tool. Another time it computed `9**9**9` itself after the calculator refused, so the tool's error message now says not to.

## Evaluation

`eval/questions.yaml` has 12 questions: 3 calculator, 4 report search, 3 that need search followed by calculation, and 2 out of scope. The main metric is **tool selection accuracy**: did the agent use exactly the expected tools?

```bash
python3 evaluate.py --check     # verify report figures appear in the index (no API calls)
python3 evaluate.py             # full run, 2 to 4 API calls per question
```

Results are written to `eval/results/`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q tests
```

65 tests in under a second, no API key: the calculator (including `9**9**9`, code injection, complex results), hybrid search and RRF, and the agent loop against a scripted Gemini (direct answers, tool chains, parallel calls, text before a call, the step limit, unknown tools, bad arguments, failover mid-run, model cooldowns, the calculator verifier, empty replies), the evaluator's scoring, display escaping, plus the API.

## Run it

```bash
pip install -r requirements.txt
echo "GEMINI_API_KEY=your-key" > .env
streamlit run streamlit_app.py          # chat UI
uvicorn app:app --port 8000             # API: POST /ask {"question": "..."}
```

Docker (API, non-root, reranker baked in, health check on `/health`):

```bash
docker build -t bi-agent .
docker run -p 8000:8000 --env-file .env bi-agent
```

## Files

| File | Purpose |
|---|---|
| `agent.py` | Agent loop, system prompt, model failover, limits |
| `tools.py` | Calculator, search tool, schemas, validated execution |
| `retrieval.py` | Hybrid search, RRF, reranker, page lookup |
| `service.py` | Lazily built client, index and embedder (nothing loads at import) |
| `app.py` | FastAPI service |
| `streamlit_app.py` | Chat UI with live tool calls |
| `display.py` | Keeps maths in questions and answers literal (`9**9**9`, `$`) when rendered as markdown |
| `evaluate.py`, `eval/questions.yaml` | Agent evaluation |
| `tests/` | Unit tests |
| `data/chunk_pages.json` | Chunk to PDF page map |
| `chroma_db/` | Prebuilt vector index |

## Limitations

- 82 of 612 pages are indexed; financial statement tables are not.
- Retrieval returns 500 character chunks without parent context (the RAG project adds parent expansion).
- Conversation memory is the last 3 exchanges as plain text.
- One process, no authentication: a demo, not a multi-tenant service.

## Author

Chanindu Dahanayake
