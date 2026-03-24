from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "esg_portfolio"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Gemini
    GEMINI_API_KEY: str = ""

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000"

    # Auth (JWT)
    JWT_SECRET_KEY: str = "dev-secret-key-change-in-production"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # DRL
    MODEL_CHECKPOINT_DIR: str = "./model_checkpoints"

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

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
