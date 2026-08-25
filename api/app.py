"""
api/app.py
-----------
Entrypoint for the MATZ AI Knowledge Assistant API.

Run with (from Backend/):
    uvicorn api.app:app --reload

Wires together config, routes, and startup/shutdown lifecycle.
Route logic lives in api/routes/*, actual behavior lives in api/controllers/*.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.src.utils.logger import logger
from agent.src.utils.supabase_client import check_supabase_connection
from api.helpers.qdrant_helper import ensure_qdrant_indexes
from api.config.settings import (
    APP_TITLE, APP_DESCRIPTION, APP_VERSION,
    CORS_ALLOW_ORIGINS, CORS_ALLOW_METHODS, CORS_ALLOW_HEADERS,
)
from api.routes import (
    health_routes, overview_routes, assistant_routes,
    knowledge_routes, collection_routes, settings_routes,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MATZ API starting up...")
    ensure_qdrant_indexes()
    check_supabase_connection()
    yield
    logger.info("MATZ API shutting down...")


app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

app.include_router(health_routes.router)
app.include_router(overview_routes.router)
app.include_router(assistant_routes.router)
app.include_router(knowledge_routes.router)
app.include_router(collection_routes.router)
app.include_router(settings_routes.router)