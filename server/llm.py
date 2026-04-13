from __future__ import annotations

import json
import httpx

from server.config import settings
from server.rag import SYSTEM_PROMPT, VOICE_SYSTEM_PROMPT, format_context  # noqa: F401

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.3-70b-versatile"


def _build_rag_prompt(user_message: str, context: str, abstained: bool) -> str:
    if abstained or not context.strip():
        return (
            f"User asked:\n{user_message}\n\n"
            "There is NO relevant context retrieved for this question. "
            "Respond honestly in 1-2 sentences. Suggest 2-3 related topics from "
            "Pooja's background (skills, projects, experience) that you can speak to."
        )
    return (
        f"CONTEXT (retrieved from Pooja's resume and GitHub):\n{context}\n\n"
        f"User question: {user_message}\n\n"
        "Answer using ONLY the CONTEXT above. Be specific — use exact names, tech, metrics."
    )


async def groq_call(messages: list[dict], tools: list | None = None) -> dict:
    """Non-streaming Groq call — used for tool detection."""
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    body: dict = {
        "model": _GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(_GROQ_URL, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()


async def stream_groq_messages(messages: list[dict]):
    """Stream tokens from an arbitrary messages list."""
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": _GROQ_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", _GROQ_URL, headers=headers, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    token = chunk["choices"][0]["delta"].get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError):
                    continue


async def stream_chat_answer(
    user_message: str,
    context: str,
    *,
    abstained: bool,
    for_voice: bool = False,
):
    """RAG path: build prompt from context and stream response."""
    prompt = _build_rag_prompt(user_message, context, abstained)
    system = VOICE_SYSTEM_PROMPT if for_voice else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    async for token in stream_groq_messages(messages):
        yield token


async def generate_chat_once(
    user_message: str,
    context: str,
    *,
    abstained: bool,
    for_voice: bool = False,
) -> str:
    parts: list[str] = []
    async for t in stream_chat_answer(user_message, context, abstained=abstained, for_voice=for_voice):
        parts.append(t)
    return "".join(parts)
