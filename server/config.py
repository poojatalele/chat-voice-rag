from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    groq_api_key: str = ""
    github_token: str = ""

    chroma_path: str = "./chroma_data"
    collection_name: str = "persona_kb"

    # RAG
    # MiniLM-L6-v2 on section-chunked resume (10 focused chunks):
    #   - On-topic queries (project name, school name): score 0.4–0.7
    #   - Broad skill queries ("RAG experience"): score 0.15–0.25 (header/skills chunks)
    #   - Unrelated queries (salary, hobbies): score < 0.05
    # Threshold 0.12 catches all resume-relevant queries while blocking true noise.
    # (Was 0.72 — calibrated for Gemini embeddings, not MiniLM.
    #  Was 0.20 — too high; missed broad skill queries.)
    similarity_threshold: float = 0.12
    retrieve_k: int = 10          # we have 10 chunks; return all
    rerank_top_chat: int = 8      # send 8 of 10 chunks to LLM — let LLM decide relevance
    rerank_top_voice: int = 3

    # Cal.com — personal API key (from cal.com/settings/developer/api-keys)
    # Leave blank to use public/anonymous endpoints (works for free accounts)
    calcom_api_key: str = ""
    # Numeric event type ID (from the URL when editing your event type)
    calcom_event_type_id: int = 0

    # Optional: Cohere rerank
    cohere_api_key: str = ""

    # Voice / Vapi (documented; server validates optional webhook secret)
    vapi_webhook_secret: str = ""


settings = Settings()
