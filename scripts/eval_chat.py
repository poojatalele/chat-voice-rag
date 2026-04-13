#!/usr/bin/env python3
"""
Run 20 scripted chat queries against /api/chat; write artifacts/eval_run.csv for manual labeling.
Usage:
  python scripts/eval_chat.py --base-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "eval" / "questions.json",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only first 5 questions",
    )
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.smoke:
        questions = questions[:5]

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "eval_run.csv"

    base = args.base_url.rstrip("/")
    rows = []
    with httpx.Client(timeout=120.0) as client:
        for q in questions:
            t0 = time.perf_counter()
            r = client.post(
                f"{base}/api/chat",
                json={"messages": [{"role": "user", "content": q["text"]}]},
            )
            ms = int((time.perf_counter() - t0) * 1000)
            if r.status_code != 200:
                rows.append(
                    {
                        "id": q["id"],
                        "expected": q.get("expected", ""),
                        "response": "",
                        "abstained": "",
                        "max_score": "",
                        "top_chunks": "",
                        "latency_ms": ms,
                        "http_error": r.status_code,
                        "manual_label": "",
                    }
                )
                continue
            data = r.json()
            cites = data.get("citations") or []
            top = json.dumps(cites[:3], ensure_ascii=False)
            rows.append(
                {
                    "id": q["id"],
                    "expected": q.get("expected", ""),
                    "response": (data.get("reply") or "")[:2000],
                    "abstained": data.get("abstained"),
                    "max_score": data.get("max_score"),
                    "top_chunks": top,
                    "latency_ms": ms,
                    "http_error": "",
                    "manual_label": "",
                }
            )

    fieldnames = [
        "id",
        "expected",
        "response",
        "abstained",
        "max_score",
        "top_chunks",
        "latency_ms",
        "http_error",
        "manual_label",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote {out_path} ({len(rows)} rows). Add manual_label column for hallucination audit.")


if __name__ == "__main__":
    main()
