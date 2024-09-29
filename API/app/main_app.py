import logging

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.logger import logger as fastapi_logger

# custom libs
from app.routers.user_access import user_access_router
from app.routers.manage_calculate import manage_calculate_routers
from app.helpers.verify_token import verify_token
from app.config.settings import get_settings

settings = get_settings()

# setup logger
gunicorn_error_logger = logging.getLogger("gunicorn.error")
gunicorn_logger = logging.getLogger("gunicorn")
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers = gunicorn_error_logger.handlers
fastapi_logger.handlers = gunicorn_error_logger.handlers


app = FastAPI(
    title="API доступа к получению виральных видео",
    description="Сделано в рамках хакатона",
    version="0.0.1",
    contact={
        "name": "Maksim Kulagin",
        "email": "maksimkulagin06@yandex.ru",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['POST', 'GET'],
    allow_headers=['*'],
    allow_credentials=True,
)

# users
app.include_router(
    user_access_router, prefix='/users',
    tags=['user_access_router']
)

app.include_router(
    manage_calculate_routers, prefix='/upload',
    tags=['manage_calculate_routers'],
    dependencies=[Depends(verify_token)]
)