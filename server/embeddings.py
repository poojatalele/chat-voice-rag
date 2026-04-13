from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Local all-MiniLM-L6-v2 via ONNX — no API key required, 384-dim embeddings.
_ef = DefaultEmbeddingFunction()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of documents using the local MiniLM model."""
    return [v.tolist() for v in _ef(texts)]


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return _ef([text])[0].tolist()
