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


def token_split(text: str, max_tokens: int = 200, overlap: int = 40) -> list[str]:
    """
    Split text into focused chunks (~200 tokens / ~800 chars) with overlap.

    Why 200 tokens:
      MiniLM-L6-v2 (the local embedding model) produces cosine similarities in
      the 0.05–0.40 range for resume-style Q&A. Long 500-token chunks dilute the
      embedding vector across many unrelated topics, pushing all similarities toward
      zero. 200-token chunks keep each chunk focused on one topic (education, one
      project, one role), lifting relevant similarities to 0.25–0.50 and making the
      threshold gate meaningful.

    Overlap of 40 tokens preserves cross-paragraph context without re-embedding
    the same information redundantly.
    """
    max_chars = max_tokens * 4       # ~4 chars per token
    overlap_chars = overlap * 4
    chunks: list[str] = []
    paras = re.split(r"\n\n+", text)
    buf = ""
    for p in paras:
        if not p.strip():
            continue
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
            # carry overlap into the next chunk
            buf = buf[-overlap_chars:].strip() + "\n\n" + p if buf else p
        while len(buf) > max_chars:
            chunks.append(buf[:max_chars])
            buf = buf[max_chars - overlap_chars:]
    if buf.strip():
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


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


_SECTION_HEADERS = re.compile(
    r"^(Education|Experience|Projects?|Technical Skills?|Skills?|"
    r"Certifications?|Awards?|Publications?|Interests?)$",
    re.IGNORECASE,
)

# Bullet chars that PDFs emit (•, ◆, ▪, and PDF font substitution artifacts)
_BULLET = re.compile(r"^[\u2022\u25c6\u25aa\uf0b7\uf0a7\uf076\u2013\-]")


def _is_entry_title(line: str) -> bool:
    """Heuristic: a non-bullet line that's short and title-cased → new entry."""
    stripped = line.strip()
    if not stripped or _BULLET.match(stripped):
        return False
    if _SECTION_HEADERS.match(stripped):
        return True
    # Project names / job titles: short, no trailing punctuation, mixed case
    if len(stripped) < 80 and not stripped.endswith((".","?","!")) and " " in stripped:
        return True
    return False


def ingest_resume(path: Path) -> list[dict]:
    """
    Section-aware resume ingestion.

    PDF extractors collapse all newlines to single \\n, so generic paragraph
    splitting (double newline) produces only 3–6 giant chunks. This function
    instead groups lines into logical entries (one education institution, one
    job role, one project) so each chunk stays focused — improving MiniLM
    embedding alignment with short queries like "where did you study?".
    """
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            raw = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF: {e}") from e
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")

    lines = [l for l in raw.splitlines() if l.strip()]

    # Group lines into logical entries separated by section headers / entry titles
    groups: list[tuple[str, list[str]]] = []   # (section_label, lines)
    current_section = "header"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if _SECTION_HEADERS.match(stripped):
            if current_lines:
                groups.append((current_section, current_lines))
            current_section = stripped.lower()
            current_lines = [stripped]
        elif _is_entry_title(stripped) and current_section not in ("header",):
            # New entry inside a section — start a new chunk
            if current_lines:
                groups.append((current_section, current_lines))
            current_lines = [stripped]
        else:
            current_lines.append(stripped)

    if current_lines:
        groups.append((current_section, current_lines))

    # Merge consecutive groups in the same section if the first is tiny (<120 chars)
    merged: list[tuple[str, str]] = []
    for section, g_lines in groups:
        text = "\n".join(g_lines).strip()
        if not text:
            continue
        if merged and merged[-1][0] == section and len(merged[-1][1]) < 120:
            merged[-1] = (section, merged[-1][1] + "\n" + text)
        else:
            merged.append((section, text))

    # Convert each group to document chunks; further split large groups
    docs: list[dict] = []
    chunk_idx = 0
    for section, text in merged:
        # Skip chunks that are too small to be useful embeddings
        if len(text.strip()) < 60:
            continue
        sub_chunks = token_split(text) if len(text) > 800 else [text]
        for chunk in sub_chunks:
            if len(chunk.strip()) < 60:
                continue
            docs.append({
                "text": chunk,
                "meta": {
                    "source": "resume",
                    "repo_name": "",
                    "file_path": str(path.name),
                    "section": section,
                    "chunk_index": chunk_idx,
                },
            })
            chunk_idx += 1

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
