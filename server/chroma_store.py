from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from server.config import settings


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=settings.chroma_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_or_create_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    client = get_client()
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    get_or_create_collection()
