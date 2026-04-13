from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from server import calendar_calcom
from server.config import settings
from server.llm import (
    SYSTEM_PROMPT,
    VOICE_SYSTEM_PROMPT,
    generate_chat_once,
    groq_call,
    stream_chat_answer,
    stream_groq_messages,
)
from server.rag import chunks_to_citations, format_context, retrieve
from server.tools import GROQ_TOOLS, execute_tool, has_scheduling_intent

app = FastAPI(title="Scaler AI Persona API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "persona-api"}


# ── RAG debug endpoint ────────────────────────────────────────────────────────

class RetrieveBody(BaseModel):
    query: str
    for_voice: bool = False
    conversation_tail: str | None = None


@app.post("/rag/retrieve")
async def rag_retrieve(body: RetrieveBody):
    chunks, max_score = retrieve(
        body.query,
        for_voice=body.for_voice,
        conversation_tail=body.conversation_tail,
    )
    abstained = max_score < settings.similarity_threshold
    return {
        "max_score": round(max_score, 4),
        "abstained": abstained,
        "chunks": [
            {"id": c.id, "text": c.text, "score": round(c.score, 4), "meta": c.meta}
            for c in chunks
        ],
    }


# ── Shared agent core ─────────────────────────────────────────────────────────

async def _agent_stream(
    last_user: str,
    history: list[dict],          # conversation WITHOUT system message
    *,
    for_voice: bool = False,
    conversation_tail: str | None = None,
) -> AsyncIterator[tuple[str, object]]:
    """
    Single source of truth for agent logic — used by both chat and voice.

    Yields (event_type, payload) tuples:
      ("token",  str)       — a text token to append
      ("slots",  list)      — slot data for chat UI (suppressed in voice mode)
      ("done",   dict)      — completion metadata {abstained, confidence, citations}
    """
    system = VOICE_SYSTEM_PROMPT if for_voice else SYSTEM_PROMPT

    # ── Scheduling path: Groq tool calling ───────────────────────────────────
    if has_scheduling_intent(last_user):
        messages = [{"role": "system", "content": system}] + history

        response = await groq_call(messages, tools=GROQ_TOOLS)
        assistant_msg = response["choices"][0]["message"]
        tool_calls = assistant_msg.get("tool_calls") or []

        if tool_calls:
            messages.append(assistant_msg)
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"] or "{}")
                tool_result, slots_data = await execute_tool(fn_name, fn_args)

                # Emit slot cards only for chat UI — voice lets LLM speak the JSON
                if fn_name == "get_availability" and slots_data and not for_voice:
                    yield ("slots", slots_data)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

            async for token in stream_groq_messages(messages):
                yield ("token", token)

        else:
            # LLM declined tools — stream its plain text reply
            text = assistant_msg.get("content") or ""
            for word in text.split(" "):
                yield ("token", word + " ")

        yield ("done", {"abstained": False, "confidence": 1.0, "citations": []})
        return

    # ── RAG path: retrieve → rerank → stream ─────────────────────────────────
    chunks, max_score = retrieve(
        last_user, for_voice=for_voice, conversation_tail=conversation_tail
    )
    abstained = max_score < settings.similarity_threshold
    ctx = format_context(chunks) if not abstained else ""

    async for token in stream_chat_answer(
        last_user, ctx, abstained=abstained, for_voice=for_voice
    ):
        yield ("token", token)

    yield ("done", {
        "abstained": abstained,
        "confidence": round(max_score, 4),
        "citations": [] if abstained else chunks_to_citations(chunks),
    })


# ── Chat endpoints ────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    for_voice: bool = False


def _parse_chat_body(messages: list[ChatMessage]) -> tuple[str, list[dict], str | None]:
    """Return (last_user_msg, history_dicts, conversation_tail)."""
    if not messages:
        return "", [], None
    user_msgs = [m for m in messages if m.role == "user"]
    if not user_msgs:
        return "", [], None
    last = user_msgs[-1].content
    history = [{"role": m.role, "content": m.content} for m in messages]
    tail_parts = [f"{m.role}: {m.content}" for m in messages[-5:]]
    tail = "\n".join(tail_parts) if len(messages) > 1 else None
    return last, history, tail


@app.post("/api/chat")
async def chat_once(body: ChatBody):
    """Non-streaming chat — RAG only (no tool calling)."""
    last, _, tail = _parse_chat_body(body.messages)
    if not last:
        raise HTTPException(400, "No user message")
    chunks, max_score = retrieve(last, for_voice=False, conversation_tail=tail)
    abstained = max_score < settings.similarity_threshold
    ctx = format_context(chunks) if not abstained else ""
    text = await generate_chat_once(last, ctx, abstained=abstained)
    return {
        "reply": text,
        "abstained": abstained,
        "max_score": round(max_score, 4),
        "citations": [] if abstained else chunks_to_citations(chunks),
    }


async def _sse_chat(body: ChatBody) -> AsyncIterator[bytes]:
    last, history, tail = _parse_chat_body(body.messages)
    if not last:
        yield f"event: error\ndata: {json.dumps({'error': 'no user message'})}\n\n".encode()
        return

    yield f"event: meta\ndata: {json.dumps({'request_id': str(uuid.uuid4())})}\n\n".encode()

    async for event_type, data in _agent_stream(last, history, for_voice=False, conversation_tail=tail):
        if event_type == "token":
            yield f"event: token\ndata: {json.dumps({'t': data})}\n\n".encode()
        elif event_type == "slots":
            yield f"event: slots\ndata: {json.dumps({'slots': data})}\n\n".encode()
        elif event_type == "done":
            yield f"event: done\ndata: {json.dumps(data)}\n\n".encode()


@app.get("/api/chat/stream")
async def chat_stream_get():
    raise HTTPException(405, "Use POST /api/chat/stream with JSON body")


@app.post("/api/chat/stream")
async def chat_stream_post(body: ChatBody):
    return StreamingResponse(_sse_chat(body), media_type="text/event-stream")


# ── Calendar endpoints ────────────────────────────────────────────────────────

class AvailabilityWindowBody(BaseModel):
    date: str        # "YYYY-MM-DD"
    start: str       # "HH:MM" in recruiter's timezone
    end: str         # "HH:MM"
    timezone: str = "UTC"


@app.post("/api/availability")
async def get_availability(body: AvailabilityWindowBody):
    slots = await calendar_calcom.fetch_slots_in_window(
        date=body.date,
        start_time=body.start,
        end_time=body.end,
        timezone_name=body.timezone,
    )
    return {"slots": slots, "timezone": body.timezone}


class BookRequestBody(BaseModel):
    start_iso: str
    attendee_name: str
    attendee_email: str
    timezone: str = "UTC"


@app.post("/api/book")
async def book_slot(body: BookRequestBody):
    result = await calendar_calcom.create_booking(
        start_iso=body.start_iso,
        attendee_name=body.attendee_name,
        attendee_email=body.attendee_email,
        timezone_name=body.timezone,
    )
    if not result.get("ok"):
        raise HTTPException(502, detail=result)
    return result


# ── Retell AI — Custom LLM WebSocket endpoint ────────────────────────────────
#
# Retell Dashboard setup (one-time):
#   1. LLMs → Add LLM → Custom LLM
#      WebSocket URL: wss://<your-render-app>.onrender.com/llm-websocket
#   2. Agents → Add Agent → select that LLM → pick a voice
#      Set interruption_sensitivity ~0.7, responsiveness ~0.8
#   3. Phone Numbers → Buy Number → assign the agent
#
# Protocol: Retell sends JSON over WS, we stream JSON chunks back.
# Reuses _agent_stream() — same RAG + tool-calling as chat, voice-optimized.

_VOICE_GREETING = (
    "Hi! I'm Pooja's AI representative. I can tell you about her background, "
    "projects, and experience, or help you schedule an interview. "
    "What would you like to know?"
)


async def _send_retell(ws: WebSocket, response_id: int, content: str, complete: bool) -> None:
    await ws.send_json({
        "response_id": response_id,
        "content": content,
        "content_complete": complete,
        "end_call": False,
    })


@app.websocket("/llm-websocket/{call_id}")
async def retell_websocket(ws: WebSocket, call_id: str):
    """
    Retell Custom LLM WebSocket handler.
    Same _agent_stream() as /api/chat/stream — no code duplication.
    """
    await ws.accept()

    # Retell expects the agent to speak first
    await _send_retell(ws, 0, _VOICE_GREETING, complete=True)

    try:
        async for raw in ws.iter_text():
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                continue

            interaction_type = req.get("interaction_type", "")
            response_id = req.get("response_id", 0)
            transcript: list[dict] = req.get("transcript", [])

            # update_only = live transcript update, no response needed
            if interaction_type == "update_only":
                continue

            # Normalize Retell "agent" role → OpenAI "assistant"
            history = [
                {
                    "role": "assistant" if m["role"] == "agent" else m["role"],
                    "content": m["content"],
                }
                for m in transcript
            ]

            if interaction_type == "reminder_required":
                # User went silent — inject a gentle nudge and skip full RAG
                history.append({
                    "role": "user",
                    "content": "(User hasn't spoken. Ask gently if they have a question about Pooja.)",
                })
                last = "__reminder__"
            else:
                last = next(
                    (m["content"] for m in reversed(transcript) if m["role"] == "user"),
                    "",
                )

            # Stream response — same agent core as chat
            async for event_type, data in _agent_stream(last, history, for_voice=True):
                if event_type == "token":
                    await _send_retell(ws, response_id, data, complete=False)
                # "slots" events dropped — LLM speaks them as plain text in voice mode

            # Signal turn complete
            await _send_retell(ws, response_id, "", complete=True)

    except WebSocketDisconnect:
        pass  # caller hung up — normal exit


# ── Exception handler ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exc(_, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── React frontend — must be LAST so API routes take priority ─────────────────

_dist = Path(__file__).resolve().parents[1] / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
