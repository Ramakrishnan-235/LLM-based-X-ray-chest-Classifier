from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    postgres_user: str = "cxr_admin"
    postgres_password: str = "cxr_local_secret"
    postgres_db: str = "cxr_insight"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    redis_host: str = "localhost"
    redis_port: int = 6379

    minio_root_user: str = "minio_admin"
    minio_root_password: str = "minio_local_secret"
    minio_bucket_name: str = "cxr-storage"
    minio_endpoint: str = "http://localhost:9000"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
