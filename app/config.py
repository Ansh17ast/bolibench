"""Application configuration management using Pydantic Settings."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """BoliBench environment and API configuration."""

    # Gnani Prisma API Credentials
    gnani_api_key: Optional[str] = None
    gnani_base_url: str = "https://api.vachana.ai/stt/v3"
    gnani_request_timeout_seconds: float = 30.0

    # Administrative Secret for Trusted Speaker Management
    admin_api_key: Optional[str] = None

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

    # Persistent Storage Data Root (Railway Volume Mount, e.g. /data)
    bolibench_data_dir: Optional[str] = None

    # Production Database Safeguard: If True, halts startup if expected DB does not exist
    bolibench_require_existing_db: bool = False

    # Allowed CORS Origins (comma-separated, e.g. https://my-app.up.railway.app)
    allowed_origins: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def data_dir_path(self) -> Optional[Path]:
        """Returns resolved Path to persistent data directory if configured."""
        env_val = self.bolibench_data_dir or os.environ.get("BOLIBENCH_DATA_DIR")
        if env_val and env_val.strip():
            return Path(env_val.strip()).resolve()
        return None

    @property
    def is_api_configured(self) -> bool:
        """Returns True if a real Gnani API key is configured."""
        if not self.gnani_api_key:
            return False
        key = self.gnani_api_key.strip()
        if not key or key == "your_gnani_api_key_here":
            return False
        return True

    @property
    def is_admin_configured(self) -> bool:
        """Returns True if an admin API key is configured."""
        if not self.admin_api_key:
            return False
        key = self.admin_api_key.strip()
        if not key or key == "your_admin_secret_key_here":
            return False
        return True

    def __repr__(self) -> str:
        """Safe string representation preventing credential leaks."""
        return (
            f"<Settings app_env={self.app_env} gnani_api_configured={self.is_api_configured} "
            f"admin_configured={self.is_admin_configured} "
            f"live_eval_enabled={self.live_evaluation_enabled} live_transcription_enabled={self.live_transcription_enabled} "
            f"data_dir={self.bolibench_data_dir}>"
        )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


def get_data_dir() -> Optional[Path]:
    """Returns the configured persistent data root directory (e.g. /data on Railway)."""
    return get_settings().data_dir_path


def get_recordings_dir() -> Path:
    """Resolves the active human recordings directory."""
    env_dir = os.environ.get("BOLIBENCH_RECORDINGS_DIR")
    if env_dir and env_dir.strip():
        return Path(env_dir.strip()).resolve()
    data_dir = get_data_dir()
    if data_dir is not None:
        return data_dir / "human_recordings_v1"
    return Path(__file__).resolve().parent.parent / "human_recordings_v1"


def get_benchmark_results_dir() -> Path:
    """Resolves the persistent benchmark results directory."""
    data_dir = get_data_dir()
    if data_dir is not None:
        return data_dir / "benchmark_results"
    return Path(__file__).resolve().parent.parent / "benchmark_results"


def get_benchmark_checkpoints_dir() -> Path:
    """Resolves the persistent benchmark checkpoints directory."""
    data_dir = get_data_dir()
    if data_dir is not None:
        return data_dir / "benchmark_checkpoints"
    return Path(__file__).resolve().parent.parent / "benchmark_checkpoints"
