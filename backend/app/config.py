"""
ResumeIQ Configuration Module.
Loads environment variables safely using Pydantic Settings.
"""

from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration and environment settings."""

    # Application Info
    APP_NAME: str = "ResumeIQ"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Cross-Origin Resource Sharing (CORS)
    CORS_ORIGINS: str = (
        "http://localhost:3000,http://localhost:5500,http://127.0.0.1:5500,http://127.0.0.1:8000"
    )

    # AI & Model Configuration
    # GEMINI_API_KEY is optional initially and will be provided at final integration
    GEMINI_API_KEY: str = ""
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

    # Pydantic Settings Config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Convert comma-separated CORS_ORIGINS string into a clean list of origins."""
        if not self.CORS_ORIGINS:
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached instance of the Settings.
    Ensures environment variables are parsed once and reused across requests.
    """
    return Settings()

