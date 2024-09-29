from pydantic_settings import BaseSettings
from pydantic import Extra
import os


ENV_API = os.getenv("ENVIRONMENT")


class Settings(BaseSettings):
    # user
    secret_key: str
    algorithm: str
    access_token_expires_hours: int

    # app config
    path_to_files: str
    path_weights_whisper: str
    device: str
    url_llama_model: str
    user_login_llama: str
    password_login_llama: str

    class Config:
        env_file = ".env" if not ENV_API else f".env.{ENV_API}"
        extra = Extra.ignore


def get_settings():
    return Settings()
