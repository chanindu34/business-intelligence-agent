"""FastAPI service for the Business Intelligence Agent.

    uvicorn app:app --port 8000
    curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question": "What is 128 / 4?"}'
"""

from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent import run_agent
from service import get_client, get_embedder, get_index

app = FastAPI(title="Business Intelligence Agent")


class Message(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    history: Optional[List[Message]] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QuestionRequest) -> Dict:
    result = run_agent(
        request.question.strip(),
        get_client(),
        index=get_index(),
        embed=get_embedder(),
        history=[m.model_dump() for m in request.history or []],
    )
    return {
        "answer": result.answer,
        "tools_used": result.tools_used,
        "pages": result.pages,
        "model": result.model,
        "turns": result.turns,
        "stopped": result.stopped,
        "seconds": result.seconds,
        "steps": [{"tool": s["tool"], "args": s["args"], "success": s["result"].get("success")} for s in result.steps],
    }
