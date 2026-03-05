"""FastAPI streaming wrapper for AIAgent.

Exposes hermes-agent as an HTTP service with SSE streaming.
Run locally with: uvicorn serve:app --host 0.0.0.0 --port 8000
Deploy on Modal via modal_app.py.
"""

import asyncio
import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# Force HERMES_HOME to a writable path. Modal secrets may set HERMES_HOME to
# a non-existent path (e.g. /app/tinker-atropos) — override unconditionally.
_hermes_home = Path("/tmp/hermes")
_hermes_home.mkdir(parents=True, exist_ok=True)
(_hermes_home / "logs").mkdir(parents=True, exist_ok=True)
os.environ["HERMES_HOME"] = str(_hermes_home)

# Pre-import modules that register signal handlers so they run in the
# main thread (signal.signal() fails if called from a worker thread).
try:
    import tools.browser_tool  # noqa: F401
except Exception:
    pass

try:
    from run_agent import AIAgent  # noqa: F401
except Exception as e:
    logger.warning("Failed to pre-import AIAgent: %s", e)

app = FastAPI(title="hermes-agent", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/agent/stream")
async def agent_stream(request: Request):
    body = await request.json()

    messages = body.get("messages", [])
    model = body.get("model", "anthropic/claude-opus-4.6")
    system_prompt = body.get("system_prompt")
    toolsets = body.get("toolsets")
    max_iterations = body.get("max_iterations", 30)
    base_url = body.get("base_url") or os.getenv("AGENT_LLM_BASE_URL")
    api_key = body.get("api_key") or os.getenv("AGENT_LLM_API_KEY")
    tags = body.get("tags")

    user_message = ""
    conversation_history = []
    for msg in messages:
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
        conversation_history.append(msg)

    if conversation_history and conversation_history[-1].get("role") == "user":
        user_message = conversation_history.pop().get("content", "")

    eq: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=512)

    def run_agent():
        try:
            agent = AIAgent(
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_iterations=max_iterations,
                quiet_mode=True,
                enabled_toolsets=toolsets,
                event_queue=eq,
                ephemeral_system_prompt=system_prompt,
                extra_tags=tags,
            )
            result = agent.run_conversation(
                user_message=user_message,
                conversation_history=conversation_history or None,
            )
            if result and result.get("failed"):
                eq.put({"type": "error", "error": result.get("error", "Agent failed")})
                eq.put({"type": "done"})
        except Exception as e:
            logger.exception("Agent error")
            eq.put({"type": "error", "error": str(e)})
            eq.put({"type": "done"})

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()

    async def event_generator():
        while True:
            try:
                event = await loop.run_in_executor(None, lambda: eq.get(timeout=120))
            except queue.Empty:
                yield "data: {\"type\": \"done\"}\n\n"
                break

            yield f"data: {json.dumps(event)}\n\n"

            if event.get("type") == "done":
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
