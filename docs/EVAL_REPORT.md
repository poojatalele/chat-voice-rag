# AI Persona — Eval Report (one page)

_Print this file to PDF (Chrome: Print → Save as PDF). Replace bracketed numbers after you run evals._

**Candidate:** [Your Name]  
**Date:** [YYYY-MM-DD]  
**Links:** Repo [URL] · Chat [URL] · Voice [E.164]

## Voice metrics

| Metric | Target | Result |
| --- | --- | --- |
| First-response latency | < 2.0s | [p50 / p90 from n=10–15 calls] |
| End-to-end latency | < 4.0s | […] |
| Interruption recovery | 10 tests, 0 crashes | [pass/fail] |
| Task completion (book) | > 80% (n=10) | [x/10] |
| Transcription (WER) | ≤ 5% WER on n=20 | [or qualitative note] |

**Method:** server logs per `call_id` + Vapi analytics (where available).

## Chat metrics

| Metric | Target | Result |
| --- | --- | --- |
| Hallucination rate | < 10% (n=20) | [% after manual_label in `artifacts/eval_run.csv`] |
| Top-3 relevance | > 85% (answerable 10) | [%] |
| Citation accuracy | > 90% | [%] |
| Factual vs cited spans | > 95% | [%] |

**Method:** `python scripts/eval_chat.py --base-url https://...` → CSV → human labels.

## Three failure modes (problem → fix → result)

1. **[Retrieval miss]** → [rerank / hybrid / threshold] → [before/after metric]
2. **[Voice latency]** → [top-3 chunks + warm `/health`] → [p90 TTFA]
3. **[Timezone]** → [explicit IANA in book flow] → [10/10 localized confirmations]

## Two-week plan

- **Week 1:** Hybrid BM25+semantic; GitHub Action for `ingest` + `eval_chat.py` on `workflow_dispatch`.
- **Week 2:** Supabase logging of low-confidence turns; lightweight analytics dashboard.
