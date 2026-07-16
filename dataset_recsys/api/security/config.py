import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Model configuration to load from .env files
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # OIDC Authentication
    OIDC_ISSUER_URL: str = os.getenv(
        "OIDC_ISSUER_URL", "https://app.datagems.eu/oauth/realms/app"
    )
    OIDC_AUDIENCE: str = os.getenv("OIDC_AUDIENCE", "dataset-recsys-api")
    GATEWAY_API_URL: str = os.getenv(
        "GATEWAY_API_URL", "https://app.datagems.eu/gw"
    )

    @property
    def OIDC_CONFIG_URL(self) -> str:
        return f"{self.OIDC_ISSUER_URL}/.well-known/openid-configuration"

    # Database
    IdpClientSecret: str
    ROOT_PATH: str = ""


settings = Settings()