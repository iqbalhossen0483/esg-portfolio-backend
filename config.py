from pydantic import model_validator
from pydantic_settings import BaseSettings


_DEFAULT_JWT_SECRET = "dev-secret-key-change-in-production"


class Settings(BaseSettings):
    # Environment
    ENV: str = "development"

    # PostgreSQL
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "esg_portfolio"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_STATEMENT_TIMEOUT_MS: int = 30_000
    DB_POOL_RECYCLE_SECONDS: int = 1_800

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Gemini
    GEMINI_API_KEY: str = ""

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000"

    # Auth (JWT)
    JWT_SECRET_KEY: str = _DEFAULT_JWT_SECRET
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # DRL
    MODEL_CHECKPOINT_DIR: str = "./model_checkpoints"

    # Uploads
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024  # 50 MB

    @property
    def IS_DEV(self) -> bool:
        return self.ENV.lower() in ("development", "dev", "local")

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def DATABASE_URL_SYNC(self) -> str:
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def CORS_ORIGIN_LIST(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @model_validator(mode="after")
    def _enforce_prod_safety(self):
        if not self.IS_DEV and self.JWT_SECRET_KEY == _DEFAULT_JWT_SECRET:
            raise RuntimeError(
                "Refusing to start: JWT_SECRET_KEY is the default in non-development env. "
                "Set JWT_SECRET_KEY to a strong random value."
            )
        return self

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
