#!/usr/bin/env python3
"""
Ingest resume + selective GitHub files into ChromaDB with Gemini embeddings.
Re-run to refresh: python scripts/ingest.py --resume data/resume.md --repos owner/repo --reset
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import httpx  # noqa: E402

from server.chroma_store import get_or_create_collection, reset_collection  # noqa: E402
from server.config import settings  # noqa: E402
from server.embeddings import embed_texts  # noqa: E402

GITHUB_API = "https://api.github.com"

PRIORITY_FILES = [
    "README.md",
    "README.rst",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "tsconfig.json",
    "main.py",
    "app.py",
    "index.ts",
    "index.js",
    "app.ts",
    "src/main.ts",
    "src/index.ts",
]


def token_split(text: str, max_tokens: int = 500, overlap: int = 100) -> list[str]:
    """Recursive-ish split by paragraphs then size (~4 chars per token)."""
    max_chars = max_tokens * 4
    overlap_chars = overlap * 4
    chunks: list[str] = []
    paras = re.split(r"\n\n+", text)
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
            buf = p
        while len(buf) > max_chars:
            chunks.append(buf[:max_chars])
            buf = buf[max_chars - overlap_chars :]
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def gh_headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        h["Authorization"] = f"Bearer {settings.github_token}"
    return h


def fetch_file(owner: str, repo: str, path: str) -> str | None:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    r = httpx.get(url, headers=gh_headers(), timeout=30.0)
    if r.status_code != 200:
        return None
    data = r.json()
    if data.get("type") != "file":
        return None
    enc = data.get("encoding")
    content = data.get("content", "")
    if enc == "base64":
        try:
            return base64.b64decode(content).decode("utf-8", errors="replace")
        except Exception:
            return None
    return None


def fetch_commits_log(owner: str, repo: str, limit: int = 40) -> str:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
    r = httpx.get(url, headers=gh_headers(), params={"per_page": limit}, timeout=30.0)
    if r.status_code != 200:
        return ""
    lines = []
    for c in r.json():
        msg = (c.get("commit") or {}).get("message", "").split("\n")[0]
        sha = c.get("sha", "")[:7]
        lines.append(f"- {sha} {msg}")
    return "Commit history (recent):\n" + "\n".join(lines)


def ingest_github_repo(owner: str, repo: str) -> list[dict]:
    docs: list[dict] = []
    repo_name = f"{owner}/{repo}"
    seen_paths: set[str] = set()

    for path in PRIORITY_FILES:
        if path in seen_paths:
            continue
        text = fetch_file(owner, repo, path)
        if not text:
            continue
        seen_paths.add(path)
        section = path.split("/")[-1].replace(".", "_")
        for idx, chunk in enumerate(token_split(text)):
            docs.append(
                {
                    "text": chunk,
                    "meta": {
                        "source": "github",
                        "repo_name": repo_name,
                        "file_path": path,
                        "section": section,
                        "chunk_index": idx,
                    },
                }
            )

    clog = fetch_commits_log(owner, repo)
    if clog:
        for idx, chunk in enumerate(token_split(clog, max_tokens=400, overlap=80)):
            docs.append(
                {
                    "text": chunk,
                    "meta": {
                        "source": "github",
                        "repo_name": repo_name,
                        "file_path": "commit_log",
                        "section": "commit_log",
                        "chunk_index": idx,
                    },
                }
            )
    return docs


def ingest_resume(path: Path) -> list[dict]:
    text = ""
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF: {e}") from e
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    docs = []
    for idx, chunk in enumerate(token_split(text)):
        docs.append(
            {
                "text": chunk,
                "meta": {
                    "source": "resume",
                    "repo_name": "",
                    "file_path": str(path.name),
                    "section": "resume",
                    "chunk_index": idx,
                },
            }
        )
    return docs


def stable_id(meta: dict, text: str) -> str:
    raw = json.dumps(meta, sort_keys=True) + text[:200]
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument(
        "--repos",
        type=str,
        default="",
        help="Comma-separated owner/repo pairs",
    )
    parser.add_argument("--reset", action="store_true", help="Delete collection first")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY") and not settings.gemini_api_key:
        print("Set GEMINI_API_KEY in environment or .env")
        sys.exit(1)

    all_docs: list[dict] = []
    all_docs.extend(ingest_resume(args.resume))

    for pair in [p.strip() for p in args.repos.split(",") if p.strip()]:
        parts = pair.split("/")
        if len(parts) != 2:
            print(f"Skip invalid repo: {pair}")
            continue
        print(f"Ingesting {pair}...")
        all_docs.extend(ingest_github_repo(parts[0], parts[1]))

    if args.reset:
        print("Resetting Chroma collection...")
        reset_collection()

    col = get_or_create_collection()
    texts = [d["text"] for d in all_docs]
    metas = [d["meta"] for d in all_docs]
    ids = [stable_id(m, t) for m, t in zip(metas, texts)]

    print(f"Embedding {len(texts)} chunks...")
    embs = embed_texts(texts)
    col.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)
    print("Done. Collection count:", col.count())


if __name__ == "__main__":
    main()
