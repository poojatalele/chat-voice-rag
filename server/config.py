from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    groq_api_key: str = ""
    github_token: str = ""

    chroma_path: str = "./chroma_data"
    collection_name: str = "persona_kb"

    # RAG
    similarity_threshold: float = 0.72
    retrieve_k: int = 20
    rerank_top_chat: int = 5
    rerank_top_voice: int = 3

    # Cal.com
    calcom_api_key: str = ""
    calcom_base_url: str = "https://api.cal.com/v1"
    calcom_username: str = ""
    calcom_event_type_id: int = 0

    # Optional: Cohere rerank
    cohere_api_key: str = ""

    # Voice / Vapi (documented; server validates optional webhook secret)
    vapi_webhook_secret: str = ""


settings = Settings()
