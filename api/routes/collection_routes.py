"""api/routes/collection_routes.py — Collections page."""

from fastapi import APIRouter

from api.controllers.collection_controller import (
    get_collections_ctrl, create_collection_ctrl, delete_collection_ctrl,
)
from api.schemas.schemas import CollectionCreate
from api.config.settings import DEFAULT_ORGANIZATION_ID

router = APIRouter(tags=["collections"])


@router.get("/collections")
async def get_collections(organization_id: str = DEFAULT_ORGANIZATION_ID):
    return await get_collections_ctrl(organization_id)


@router.post("/collections")
async def create_new_collection(request: CollectionCreate):
    return await create_collection_ctrl(request)


@router.delete("/collections/{collection_id}")
async def delete_collection(collection_id: str):
    return await delete_collection_ctrl(collection_id)
