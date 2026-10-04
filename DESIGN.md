# Design notes

## Why an agent, not a fixed pipeline

Some questions need a search, some a calculation, some both in sequence ("what share of EBITDA came from Retail?" needs two figures, then a division), and some neither. A fixed pipeline would either always search (slow, wasteful for "128 / 4") or need hand-written routing rules. Tool calling lets the model choose, while the code keeps control of execution.

## Where control stays in code

- **Execution:** the SDK's automatic function calling is disabled; every tool call is validated with Pydantic and run by `execute_tool`, which never raises.
- **Limits:** 6 turns, 4 calls per turn, 90 s per question, calculator bounds.
- **Grounding:** the system prompt requires a search for any company fact and the calculator for any arithmetic.
- **Failover:** quota and outage handling is outside the model's view; a failed model is replaced and the run restarts, which is safe because tools are read-only.

## What I would add next

- Parent passages for retrieval, as in the RAG project.
- A check that every number in the answer appears in a tool result.
- Tracing (Langfuse or LangSmith) for production debugging.
- LangGraph once the flow needs branches like "grade passages, rewrite query, retry".
