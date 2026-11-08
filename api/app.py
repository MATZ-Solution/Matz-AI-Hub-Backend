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

from agents.langgraph_agent.utils.utils import logger
from agents.langgraph_agent.utils.utils import check_supabase_connection
from api.helpers.qdrant_helper import ensure_qdrant_indexes
from api.errors import install_error_handling
from api.auth import install_auth, seed_admin
from api.config.settings import (
    APP_TITLE, APP_DESCRIPTION, APP_VERSION, missing_required_env, ENABLE_API_DOCS,
    CORS_ALLOW_ORIGINS, CORS_ALLOW_METHODS, CORS_ALLOW_HEADERS,
)
from api.routes import (
    health_routes, auth_routes, overview_routes, assistant_routes,
    knowledge_routes, collection_routes, settings_routes,
    analytics_routes,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MATZ API starting up...")
    missing = missing_required_env()
    if missing:
        # Fail fast: an API that boots without its provider key answers every
        # chat with an error instead of refusing to deploy.
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
            + ". Set them as Fly/Render secrets or in Backend/.env."
        )
    ensure_qdrant_indexes()
    seed_admin()
    check_supabase_connection()
    yield
    logger.info("MATZ API shutting down...")


_DOCS = {"docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json"}
app = FastAPI(
    title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION, lifespan=lifespan,
    **(_DOCS if ENABLE_API_DOCS else {k: None for k in _DOCS}),
)

# Middleware order (outermost first): CORS → error handling → auth → routes.
# Auth is installed first so it sits innermost: its 401s still get a request
# id from the error layer and CORS headers from the CORS layer.
install_auth(app, docs_paths={"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"} if ENABLE_API_DOCS else set())

# Installed before CORS so CORS stays the outermost layer and error responses
# still carry CORS headers (otherwise the browser reports them as network errors).
install_error_handling(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

app.include_router(health_routes.router)
app.include_router(auth_routes.router)
app.include_router(overview_routes.router)
app.include_router(assistant_routes.router)
app.include_router(knowledge_routes.router)
app.include_router(collection_routes.router)
app.include_router(settings_routes.router)
app.include_router(analytics_routes.router)