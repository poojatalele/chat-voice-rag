from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from server import calendar_calcom
from server.config import settings
from server.llm import generate_chat_once, groq_call, stream_chat_answer, stream_groq_messages, SYSTEM_PROMPT
from server.rag import chunks_to_citations, retrieve
from server.tools import GROQ_TOOLS, execute_tool, has_scheduling_intent

app = FastAPI(title="Scaler AI Persona API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "persona-api"}


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
            {
                "id": c.id,
                "text": c.text,
                "score": round(c.score, 4),
                "meta": c.meta,
            }
            for c in chunks
        ],
    }


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    for_voice: bool = False


def _last_user_and_tail(messages: list[ChatMessage]) -> tuple[str, str | None]:
    if not messages:
        return "", None
    user_msgs = [m for m in messages if m.role == "user"]
    if not user_msgs:
        return "", None
    last = user_msgs[-1].content
    tail_parts = []
    for m in messages[-5:]:
        tail_parts.append(f"{m.role}: {m.content}")
    tail = "\n".join(tail_parts) if len(messages) > 1 else None
    return last, tail


@app.post("/api/chat")
async def chat_once(body: ChatBody):
    last, tail = _last_user_and_tail(body.messages)
    if not last:
        raise HTTPException(400, "No user message")
    chunks, max_score = retrieve(last, for_voice=body.for_voice, conversation_tail=tail)
    from server.rag import format_context

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
    last, tail = _last_user_and_tail(body.messages)
    rid = str(uuid.uuid4())
    if not last:
        yield f"event: error\ndata: {json.dumps({'error': 'no user message'})}\n\n".encode()
        return

    yield f"event: meta\ndata: {json.dumps({'request_id': rid, 'model': 'llama-3.3-70b'})}\n\n".encode()

    # ── Scheduling path: tool calling ────────────────────────────────────────
    if has_scheduling_intent(last):
        # Build conversation history for Groq
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in body.messages:
            history.append({"role": m.role, "content": m.content})

        # Non-streaming call to detect tool use
        response = await groq_call(history, tools=GROQ_TOOLS)
        choice = response["choices"][0]
        assistant_msg = choice["message"]
        tool_calls = assistant_msg.get("tool_calls") or []

        if tool_calls:
            history.append(assistant_msg)
            slots_data: list | None = None

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"] or "{}")
                tool_result, slots_data = await execute_tool(fn_name, fn_args)

                # Emit slots event so frontend can render interactive cards
                if fn_name == "get_availability" and slots_data is not None:
                    yield f"event: slots\ndata: {json.dumps({'slots': slots_data})}\n\n".encode()

                history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

            # Stream the follow-up response with tool results injected
            async for token in stream_groq_messages(history):
                yield f"event: token\ndata: {json.dumps({'t': token})}\n\n".encode()

        else:
            # Groq chose not to call tools — stream its text response
            text = assistant_msg.get("content") or ""
            for token in text.split(" "):
                yield f"event: token\ndata: {json.dumps({'t': token + ' '})}\n\n".encode()

        yield f"event: done\ndata: {json.dumps({'abstained': False, 'confidence': 1.0, 'citations': []})}\n\n".encode()
        return

    # ── RAG path: retrieve → rerank → stream ────────────────────────────────
    from server.rag import format_context
    chunks, max_score = retrieve(last, for_voice=body.for_voice, conversation_tail=tail)
    abstained = max_score < settings.similarity_threshold
    ctx = format_context(chunks) if not abstained else ""

    async for token in stream_chat_answer(last, ctx, abstained=abstained):
        yield f"event: token\ndata: {json.dumps({'t': token})}\n\n".encode()

    done = {
        "abstained": abstained,
        "confidence": round(max_score, 4),
        "citations": [] if abstained else chunks_to_citations(chunks),
    }
    yield f"event: done\ndata: {json.dumps(done)}\n\n".encode()


@app.get("/api/chat/stream")
async def chat_stream_get():
    """GET without body — use POST for browser EventSource limitation; we use POST with fetch stream instead."""
    raise HTTPException(405, "Use POST /api/chat/stream with JSON body")


@app.post("/api/chat/stream")
async def chat_stream_post(body: ChatBody):
    return StreamingResponse(_sse_chat(body), media_type="text/event-stream")


class AvailabilityWindowBody(BaseModel):
    date: str                    # "YYYY-MM-DD"
    start: str                   # "HH:MM" in recruiter's timezone
    end: str                     # "HH:MM"
    timezone: str = "UTC"


@app.post("/api/availability")
async def get_availability(body: AvailabilityWindowBody):
    """Return Pooja's free slots within the recruiter's specified window."""
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
    """Book a slot and return confirmation with Google Meet link."""
    result = await calendar_calcom.create_booking(
        start_iso=body.start_iso,
        attendee_name=body.attendee_name,
        attendee_email=body.attendee_email,
        timezone_name=body.timezone,
    )
    if not result.get("ok"):
        raise HTTPException(502, detail=result)
    return result


class VoiceTurnBody(BaseModel):
    transcript: str
    call_id: str | None = None


@app.post("/voice/turn")
async def voice_turn(body: VoiceTurnBody):
    """Vapi server webhook: return text for assistant to speak (RAG-grounded)."""
    chunks, max_score = retrieve(body.transcript, for_voice=True)
    from server.rag import format_context

    abstained = max_score < settings.similarity_threshold
    ctx = format_context(chunks) if not abstained else ""
    text = await generate_chat_once(body.transcript, ctx, abstained=abstained)
    return {
        "reply": text,
        "abstained": abstained,
        "citations": [] if abstained else chunks_to_citations(chunks),
    }


@app.exception_handler(Exception)
async def global_exc(_, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# Serve React frontend — must be LAST so API routes take priority
_dist = Path(__file__).resolve().parents[1] / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
