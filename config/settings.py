"""Centralized application settings loaded from environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = BASE_DIR / "keys" / ".env"


class Settings(BaseSettings):
    """Application configuration sourced from environment."""

    bybit_api_key: str = Field(..., alias="BYBIT_API_KEY")
    bybit_api_secret: str = Field(..., alias="BYBIT_API_SECRET")
    telegram_token: str = Field(..., alias="TELEGRAM_TOKEN")
    telegram_chat_id: str = Field(..., alias="TELEGRAM_CHAT_ID")

    class Config:
        env_file = DEFAULT_ENV_PATH
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
