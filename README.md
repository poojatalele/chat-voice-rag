# AI Hiring Screener - Pooja Talele

A public-facing AI persona of a job candidate. Recruiters can ask questions about background, projects, and skills via **text chat** or **voice call**, and book an interview without the candidate being present.

**Live demo:** `https://chat-voice-rag.onrender.com`
**Voice:** `+17406598273` (US number, Vapi AI agent)
(Note: Using a free Twilio account, so only verified numbers can call. To verify your number, contact me directly.)

---

## What it does

| Feature | How |
|---|---|
| Chat Q&A | RAG over resume + GitHub repos, streamed token by token |
| Voice Q&A | Vapi built-in LLM + resume PDF Knowledge Base |
| Interview booking | 4-step modal (UI) or voice command to Cal.com to Google Meet + email |
| Citations | Every answer links back to the exact resume section or GitHub file |
| No-match guard | Low-confidence queries get an honest redirect, never a hallucination |

---

## Architecture

```
Resume PDF + GitHub Repos
         |
         v
  Section-aware chunker -> MiniLM-L6-v2 (local) -> ChromaDB
                                                      |
                                              vector search at runtime
                                                      |
React SPA --POST /api/chat/stream (SSE)--> FastAPI --+--> Groq LLM (streaming)
           --POST /api/availability------->           ----> Cal.com v2
           --POST /api/book------------>
Vapi Voice --POST /api/vapi/book-call-->
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, uvicorn |
| LLM | Groq - llama-3.3-70b-versatile |
| Embeddings | MiniLM-L6-v2 (local ONNX, no API key) |
| Vector DB | ChromaDB (cosine similarity, persistent on disk) |
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| Voice | Vapi AI (built-in LLM + Knowledge Base) |
| Booking | Cal.com v2 REST API |

---

