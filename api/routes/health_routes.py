"""api/routes/health_routes.py — root + health check, not tied to any page."""

from fastapi import APIRouter

from api.config.settings import APP_TITLE, APP_VERSION

router = APIRouter(tags=["health"])


@router.get("/")
def root():
    return {"message": f"{APP_TITLE} API", "docs": "/docs"}


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": APP_TITLE, "version": APP_VERSION}