from __future__ import annotations

import json
import httpx

from server.config import settings
from server.rag import SYSTEM_GUARD, format_context

# Groq OpenAI-compatible endpoint
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.3-70b-versatile"

# Gemini REST streaming endpoint (v1beta supports streaming)
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent"


def _build_prompt(user_message: str, context: str, abstained: bool) -> str:
    if abstained or not context.strip():
        return (
            f"User asked:\n{user_message}\n\n"
            "There is NO relevant context in the knowledge base for this question. "
            "Reply in 2-3 sentences that you do not have that information in the materials you were given, "
            "and suggest what kind of detail they could ask instead (only generically, do not invent facts). "
            "If they want to schedule an interview, tell them they can use the booking panel or ask for available times."
        )
    return f"CONTEXT (from resume/GitHub):\n{context}\n\nUser:\n{user_message}\n\nAnswer using only CONTEXT."


async def _stream_groq(prompt: str):
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": _GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_GUARD},
            {"role": "user", "content": prompt},
        ],
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


async def _stream_gemini(prompt: str):
    import google.generativeai as genai
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY required")
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=SYSTEM_GUARD)
    resp = await model.generate_content_async(prompt, stream=True)
    async for chunk in resp:
        if chunk.text:
            yield chunk.text


async def stream_chat_answer(
    user_message: str,
    context: str,
    *,
    abstained: bool,
):
    prompt = _build_prompt(user_message, context, abstained)
    if settings.groq_api_key:
        async for token in _stream_groq(prompt):
            yield token
    else:
        async for token in _stream_gemini(prompt):
            yield token


async def generate_chat_once(
    user_message: str,
    context: str,
    *,
    abstained: bool,
) -> str:
    parts: list[str] = []
    async for t in stream_chat_answer(user_message, context, abstained=abstained):
        parts.append(t)
    return "".join(parts)
