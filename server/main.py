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
from server.llm import generate_chat_once, stream_chat_answer
from server.rag import chunks_to_citations, retrieve

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

    chunks, max_score = retrieve(last, for_voice=body.for_voice, conversation_tail=tail)
    from server.rag import format_context

    abstained = max_score < settings.similarity_threshold
    ctx = format_context(chunks) if not abstained else ""

    yield f"event: meta\ndata: {json.dumps({'request_id': rid, 'model': 'gemini-2.0-flash'})}\n\n".encode()

    async for token in stream_chat_answer(last, ctx, abstained=abstained):
        payload = json.dumps({"t": token})
        yield f"event: token\ndata: {payload}\n\n".encode()

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


class AvailabilityBody(BaseModel):
    timezone: str = "UTC"
    days_ahead: int = 14


@app.post("/api/calendar/availability")
async def calendar_availability(body: AvailabilityBody):
    slots = await calendar_calcom.fetch_slots(days_ahead=body.days_ahead, timezone_name=body.timezone)
    return {"slots": slots, "timezone": body.timezone}


class BookBody(BaseModel):
    start_iso: str
    attendee_email: str
    attendee_name: str
    timezone: str
    notes: str = ""


@app.post("/api/calendar/book")
async def calendar_book(body: BookBody):
    result = await calendar_calcom.create_booking(
        start_iso=body.start_iso,
        attendee_email=body.attendee_email,
        attendee_name=body.attendee_name,
        timezone_name=body.timezone,
        notes=body.notes,
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
