from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from server import calendar_calcom
from server.config import settings
from server.llm import generate_chat_once, stream_chat_answer
from server.rag import SYSTEM_PROMPT, chunks_to_citations, format_context, retrieve

app = FastAPI(title="Scaler AI Persona API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health

@app.get("/health")
async def health():
    return {"status": "ok", "service": "persona-api"}


# RAG debug 

class RetrieveBody(BaseModel):
    query: str
    for_voice: bool = False
    conversation_tail: str | None = None


@app.post("/rag/retrieve")
async def rag_retrieve(body: RetrieveBody):
    chunks, max_score = retrieve(body.query, conversation_tail=body.conversation_tail)
    abstained = max_score < settings.similarity_threshold
    return {
        "max_score": round(max_score, 4),
        "abstained": abstained,
        "chunks": [
            {"id": c.id, "text": c.text, "score": round(c.score, 4), "meta": c.meta}
            for c in chunks
        ],
    }


# Shared agent core

async def _agent_stream(
    last_user: str,
    history: list[dict],
    *,
    conversation_tail: str | None = None,
) -> AsyncIterator[tuple[str, object]]:
    """
    Yields (event_type, payload):
      ("token", str)  — text token
      ("done",  dict) — {abstained, confidence, citations}
    """
    chunks, max_score = retrieve(last_user, conversation_tail=conversation_tail)
    abstained = max_score < settings.similarity_threshold
    ctx = format_context(chunks) if not abstained else ""

    async for token in stream_chat_answer(last_user, ctx, abstained=abstained):
        yield ("token", token)

    yield ("done", {
        "abstained": abstained,
        "confidence": round(max_score, 4),
        "citations": [] if abstained else chunks_to_citations(chunks),
    })


# Chat endpoints

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)


def _parse_chat_body(messages: list[ChatMessage]) -> tuple[str, list[dict], str | None]:
    if not messages:
        return "", [], None
    user_msgs = [m for m in messages if m.role == "user"]
    if not user_msgs:
        return "", [], None
    last = user_msgs[-1].content
    history = [{"role": m.role, "content": m.content} for m in messages]
    tail = "\n".join(f"{m.role}: {m.content}" for m in messages[-5:]) if len(messages) > 1 else None
    return last, history, tail


@app.post("/api/chat")
async def chat_once(body: ChatBody):
    last, _, tail = _parse_chat_body(body.messages)
    if not last:
        raise HTTPException(400, "No user message")
    chunks, max_score = retrieve(last, conversation_tail=tail)
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

    async for event_type, data in _agent_stream(last, history, conversation_tail=tail):
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


# Calendar endpoints

class AvailabilityWindowBody(BaseModel):
    date: str
    start: str
    end: str
    timezone: str = "UTC"


@app.post("/api/availability")
async def get_availability(body: AvailabilityWindowBody):
    slots = await calendar_calcom.fetch_slots_in_window(
        date=body.date, start_time=body.start, end_time=body.end, timezone_name=body.timezone,
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

VAPI_SECRET = settings.vapi_secret

@app.post("/api/vapi/book-call")
async def vapi_book_call(request: Request):
    """
    Vapi tool-call webhook for the book_call function.
    Finds the first available slot in the caller's preferred window and books it.
    Cal.com handles: availability check, Google Meet creation, email invites.
    """
    # Verify request is from Vapi
    if VAPI_SECRET:
        secret = request.headers.get("x-vapi-secret", "")
        if secret != VAPI_SECRET:
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    body = await request.json()

    # Parse Vapi's tool-call payload
    tool_calls = body.get("message", {}).get("toolCalls", [])
    if not tool_calls:
        return JSONResponse({"results": []})

    tc = tool_calls[0]
    tool_call_id = tc.get("id", "")
    args = tc.get("function", {}).get("arguments", {})
    if isinstance(args, str):
        args = json.loads(args)

    name = args.get("name", "")
    email = args.get("email", "")
    preferred_date = args.get("preferred_date", "")
    time_start = args.get("preferred_time_start", "09:00")
    time_end = args.get("preferred_time_end", "18:00")
    timezone = args.get("timezone", "UTC")

    if not (name and email and preferred_date):
        return JSONResponse({"results": [{"toolCallId": tool_call_id, "result": "Missing required information. Please provide your name, email, and preferred date."}]})

    # Step 1: find available slots in the caller's window
    slots = await calendar_calcom.fetch_slots_in_window(
        date=preferred_date,
        start_time=time_start,
        end_time=time_end,
        timezone_name=timezone,
    )

    if not slots:
        return JSONResponse({"results": [{"toolCallId": tool_call_id, "result": f"No available slots on {preferred_date} between {time_start} and {time_end}. Ask the caller to suggest a different day or wider time window."}]})

    # Step 2: book the first available slot
    first_slot = slots[0]["start"]
    result = await calendar_calcom.create_booking(
        start_iso=first_slot,
        attendee_name=name,
        attendee_email=email,
        timezone_name=timezone,
    )

    if not result.get("ok"):
        return JSONResponse({"results": [{"toolCallId": tool_call_id, "result": "Booking failed. Ask the caller to try a different time slot."}]})

    from datetime import datetime
    try:
        dt = datetime.fromisoformat(result["start"].replace("Z", "+00:00"))
        day = dt.day
        hour = dt.hour % 12 or 12
        minute = f"{dt.minute:02d}"
        am_pm = "AM" if dt.hour < 12 else "PM"
        spoken_time = f"{dt.strftime('%A, %B')} {day} at {hour}:{minute} {am_pm}"
    except Exception:
        spoken_time = result.get("start", first_slot)

    meet = result.get("location", "")
    meet_text = f" The Google Meet link will be in the calendar invite." if meet else ""

    confirmation = (
        f"Done! I've booked a 30-minute interview for {spoken_time}. "
        f"A calendar invite has been sent to {email}.{meet_text} "
        f"Looking forward to speaking with you!"
    )

    return JSONResponse({"results": [{"toolCallId": tool_call_id, "result": confirmation}]})


# Exception handler

@app.exception_handler(Exception)
async def global_exc(_, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# React frontend — must be LAST

_dist = Path(__file__).resolve().parents[1] / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
