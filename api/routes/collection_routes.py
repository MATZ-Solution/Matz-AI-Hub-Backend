"""api/routes/collection_routes.py — Collections page."""

from fastapi import APIRouter, Depends

from api.controllers.collection_controller import (
    get_collections_ctrl, create_collection_ctrl, delete_collection_ctrl,
)
from api.schemas.schemas import CollectionCreate
from api.auth import CurrentUser, get_current_user

router = APIRouter(tags=["collections"])


@router.get("/collections")
def get_collections(user: CurrentUser = Depends(get_current_user)):
    return get_collections_ctrl(user.organization_id)


@router.post("/collections")
def create_new_collection(
    request: CollectionCreate,
    user: CurrentUser = Depends(get_current_user),
):
    return create_collection_ctrl(request, user.organization_id)


@router.delete("/collections/{collection_id}")
def delete_collection(
    collection_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    return delete_collection_ctrl(collection_id, user.organization_id)