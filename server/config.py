from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    groq_api_key: str = ""
    github_token: str = ""

    chroma_path: str = "./chroma_data"
    collection_name: str = "persona_kb"
    
    similarity_threshold: float = 0.12
    retrieve_k: int = 10          
    rerank_top_chat: int = 8      
    rerank_top_voice: int = 3

    calcom_api_key: str = ""
    calcom_event_type_id: int = 0

    # Voice / Vapi 
    vapi_secret: str = ""

settings = Settings()
