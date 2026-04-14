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

## Local setup

### 1. Backend

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Environment variables

Create `.env` in the project root:

```env
GROQ_API_KEY=your_groq_key

CALCOM_API_KEY=cal_live_...
CALCOM_EVENT_TYPE_ID=your_event_type_id

VAPI_SECRET=your_vapi_webhook_secret

VITE_API_BASE_URL=http://127.0.0.1:8000
```

### 3. Ingest knowledge base

Place your resume PDF at `data/resume.pdf`, then run:

```bash
python scripts/ingest.py --resume data/resume.pdf --repos poojatalele/blood-report-analysis,poojatalele/reddit-comments-automation,poojatalele/dsa-chat-assistant,poojatalele/person-and-ppe-detection --reset
```

Re-run with `--reset` any time the resume or repos change.

### 4. Start the API

```bash
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Start the frontend

```bash
cd web
npm install
npm run dev       # http://localhost:5173
```

---

## Cal.com setup

1. Go to [cal.com/settings/developer/api-keys](https://cal.com/settings/developer/api-keys) and create a personal API key
2. Create a 30-minute event type and copy the numeric ID from the URL
3. Set `CALCOM_API_KEY` and `CALCOM_EVENT_TYPE_ID` in `.env`

Cal.com handles availability, Google Meet creation, and email invites automatically.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/chat` | Single-turn chat (non-streaming) |
| POST | `/api/chat/stream` | Streaming chat via SSE |
| POST | `/api/availability` | Fetch available slots |
| POST | `/api/book` | Confirm a booking |
| POST | `/api/vapi/book-call` | Vapi voice booking webhook |
| POST | `/rag/retrieve` | Debug: inspect retrieved chunks |

---

## RAG configuration

All tunable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.12` | Below this score the AI redirects |
| `CHROMA_PATH` | `./chroma_data` | Vector DB storage path |

Threshold calibrated for MiniLM-L6-v2 on 200-token resume chunks:
- Specific queries (project name, school): score `0.40-0.68` -> answered
- Broad skill queries (RAG, Python): score `0.15-0.25` -> answered
- Unrelated queries (salary, hobbies): score `< 0.05` -> redirected

---

## Docker

```bash
docker compose up --build
```

Then run ingestion inside the container or mount `chroma_data` from host.

---

## Deployment (Render)

1. Push to GitHub
2. Create a new Web Service on Render and connect the repo
3. Set all env vars from `.env` in Render's environment settings
4. Add a persistent disk mounted at `/app/chroma_data`
5. Run ingestion once after first deploy via Render Shell
6. Add a keepalive ping to `GET /health` every 14 minutes (Render free tier sleeps after 15 min idle) using [UptimeRobot](https://uptimerobot.com)

---

## Evaluation

```bash
# Full eval (20 questions)
python scripts/eval_chat.py --base-url https://your-deployed-url

# Quick smoke test (first 5 questions)
python scripts/eval_chat.py --base-url https://your-deployed-url --smoke
```

Output: `artifacts/eval_run.csv` - fill `manual_label` column, paste metrics into `docs/EVAL_REPORT.md`.
