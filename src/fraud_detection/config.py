"""Validated application configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_prefix="FRAUD_", env_file=".env", extra="ignore")

    project_name: str = "real-time-fraud-gnn"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    random_seed: int = Field(default=42, ge=0)
    kafka_bootstrap_servers: str = "localhost:19092"
    kafka_transactions_topic: str = "transactions"
    kafka_alerts_topic: str = "fraud-alerts"
    neighbor_cap: int = Field(default=50, ge=1)
    paysim_url: str | None = None
    paysim_sha256: str | None = None


def get_settings() -> Settings:
    """Return validated settings for the current environment."""

    return Settings()
