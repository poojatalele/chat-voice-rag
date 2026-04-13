# Vapi (voice) integration

## Goal

Inbound phone → **Vapi assistant** → your HTTPS **`POST /voice/turn`** with caller transcript → JSON `{ "reply": "..." }` for TTS.

## Steps (free tier)

1. Create a Vapi assistant with **Server URL** (or custom function) pointing to `https://<your-api-host>/voice/turn`.
2. Request body (suggested minimal contract you control in Vapi “Custom Tool” or middleware):

```json
{ "transcript": "What is your experience with distributed systems?", "call_id": "optional" }
```

3. Response:

```json
{
  "reply": "spoken answer text",
  "abstained": false,
  "citations": []
}
```

4. For **calendar on voice**, add Vapi tools that call the same backend as the web UI:

- `POST /api/calendar/availability` `{ "timezone": "America/New_York" }`
- `POST /api/calendar/book` `{ "start_iso": "...", "attendee_email": "...", "attendee_name": "...", "timezone": "..." }`

5. **Latency:** keep Vapi **RAG path** to **top-3 chunks** (already enforced with `for_voice: true` in `/rag/retrieve` from your server if you wire retrieve inside `/voice/turn` — current `/voice/turn` uses `retrieve(..., for_voice=True)`).

6. **Quota:** log remaining Vapi minutes before submission; document any paid top-up in README **Cost** section.
