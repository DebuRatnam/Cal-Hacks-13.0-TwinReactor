"""FastAPI app: surrogate API, agent chat, and the scientist UI.

Run (the chat needs Anthropic credentials, e.g. ANTHROPIC_API_KEY):
    uvicorn backend.app:app --reload
then open http://127.0.0.1:8000/ for the UI (frontend/index.html).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.surrogate import LayoutError, get_service

app = FastAPI(title="PWR Neutronics Surrogate")
service = get_service


class CoreRequest(BaseModel):
    layout: list[list[int]]  # [9][9] assembly material ids (see GET /core); -1 outside the core
    rods: dict[str, float] = Field(default_factory=dict)  # bank id -> inserted fraction 0..1
    boron_ppm: float = 1000.0


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    core: CoreRequest | None = None  # the configuration on the scientist's screen


@app.on_event("startup")
def _load() -> None:
    _ = service().predictor  # load the checkpoint once, not on the first click


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/core")
def core() -> dict:
    return service().describe()


@app.post("/predict")
def predict(req: CoreRequest) -> dict:
    try:
        return service().predict(req.layout, req.rods, req.boron_ppm)
    except LayoutError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/solve")
def solve(req: CoreRequest) -> dict:
    """Reference ndiffusion solve. Not used by the UI; the agent verifies through its own tool."""
    try:
        return service().solve(req.layout, req.rods, req.boron_ppm)
    except LayoutError as e:
        raise HTTPException(422, str(e)) from e
    except RuntimeError as e:  # reference solve rejected (no convergence / balance check)
        raise HTTPException(500, f"Reference solve failed: {e}") from e


# ---------------------------------------------------------------- agent chat
# One CoreAgent (conversation history + evaluated configurations) per browser session, in memory.
MAX_SESSIONS = 50
_agents: OrderedDict[str, tuple] = OrderedDict()
_agents_lock = threading.Lock()


def _agent_for(session_id: str):
    import anthropic

    from agent.agent import CoreAgent

    with _agents_lock:
        if session_id not in _agents:
            try:
                agent = CoreAgent()
            except anthropic.AnthropicError as e:  # no credentials configured
                raise HTTPException(503, _NO_KEY) from e
            _agents[session_id] = (agent, threading.Lock())
            while len(_agents) > MAX_SESSIONS:
                _agents.popitem(last=False)
        _agents.move_to_end(session_id)
        return _agents[session_id]


_NO_KEY = (
    "The agent has no Anthropic credentials. Restart the server with ANTHROPIC_API_KEY set "
    "(export ANTHROPIC_API_KEY=... && uvicorn backend.app:app)."
)


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    import anthropic

    agent, lock = _agent_for(req.session_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "The agent is still answering the previous message.")
    try:
        if req.core is not None:
            agent.set_base(req.core.layout, req.core.rods, req.core.boron_ppm)
        calls: list[dict] = []
        reply = agent.ask(req.message, lambda name, args: calls.append({"name": name, "input": args}))
    except anthropic.AuthenticationError as e:
        raise HTTPException(503, _NO_KEY) from e
    except TypeError as e:  # SDK raises this when no credential source resolves at all
        if "authentication" not in str(e):
            raise
        raise HTTPException(503, _NO_KEY) from e
    except anthropic.RateLimitError as e:
        raise HTTPException(429, "Claude API rate limit reached; try again shortly.") from e
    except anthropic.APIConnectionError as e:
        raise HTTPException(503, "Couldn't reach the Claude API; check the network.") from e
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Claude API error {e.status_code}: {e.message}") from e
    finally:
        lock.release()
    s = agent.session
    configs = [
        {
            "label": r.label,
            "layout": r.layout,
            "rods": r.rods,
            "boron_ppm": r.boron_ppm,
            "result": r.out,
        }
        for r in (s.results[label] for label in s.new_labels)
    ]
    return {"reply": reply, "tool_calls": calls, "configs": configs}


# Scientist UI. Mounted last so the API routes above take precedence over "/".
app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True))
