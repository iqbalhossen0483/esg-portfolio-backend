from fastapi import APIRouter, Depends

from core.auth.dependencies import get_current_user
from core.response import success_response
from core.tools.sector_tools import get_sector_detail, get_sector_rankings
from db.models import User

router = APIRouter(prefix="/sectors", tags=["sectors"])


@router.get("")
async def list_sectors(
    sort_by: str = "composite",
    top_n: int = 11,
    user: User = Depends(get_current_user),
):
    """Get sector rankings."""
    data = await get_sector_rankings(sort_by=sort_by, top_n=top_n)
    return success_response(
        data=data,
        message="Sector rankings retrieved successfully",
    )


@router.get("/{sector}")
async def sector_detail(
    sector: str,
    user: User = Depends(get_current_user),
):
    """Get sector detail with top companies."""
    data = await get_sector_detail(sector=sector)
    return success_response(
        data=data,
        message="Sector detail retrieved successfully",
    )
