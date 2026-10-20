"""Application configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """BoliBench environment and API configuration."""

    # Gnani Prisma API Credentials
    gnani_api_key: Optional[str] = None
    gnani_base_url: str = "https://api.vachana.ai/stt/v3"
    gnani_request_timeout_seconds: float = 30.0

    # Upload Constraints (25 MB default safety ceiling)
    max_audio_size_bytes: int = 26_214_400

    # App Environment
    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000

    # Live Evaluation Safety Guard (Disabled by default to protect owner credits on public deployments)
    live_evaluation_enabled: bool = False

    # Live Transcription Safety Guard (Disabled by default to protect owner credits on single-audio uploads)
    live_transcription_enabled: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_api_configured(self) -> bool:
        """Returns True if a real Gnani API key is configured."""
        if not self.gnani_api_key:
            return False
        key = self.gnani_api_key.strip()
        if not key or key == "your_gnani_api_key_here":
            return False
        return True

    def __repr__(self) -> str:
        """Safe string representation preventing credential leaks."""
        return (
            f"<Settings app_env={self.app_env} gnani_api_configured={self.is_api_configured} "
            f"live_eval_enabled={self.live_evaluation_enabled} live_transcription_enabled={self.live_transcription_enabled}>"
        )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
