from fastapi import APIRouter, Depends

from core.auth.dependencies import get_current_user
from core.tools.company_tools import (
    get_best_companies,
    get_company_detail,
    search_similar_companies,
)
from db.models import User

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("")
async def list_companies(
    sector: str = None,
    min_esg: float = None,
    min_sharpe: float = None,
    top_n: int = 20,
    user: User = Depends(get_current_user),
):
    """List companies with optional filters."""
    return get_best_companies(
        sector=sector, min_esg=min_esg, min_sharpe=min_sharpe, top_n=top_n,
    )


@router.get("/{symbol}")
async def company_detail(symbol: str, user: User = Depends(get_current_user)):
    """Get full company detail."""
    return get_company_detail(symbol=symbol)


@router.get("/{symbol}/similar")
async def similar_companies(
    symbol: str,
    min_esg: float = None,
    min_sharpe: float = None,
    top_n: int = 5,
    user: User = Depends(get_current_user),
):
    """Find similar companies via pgvector search."""
    return search_similar_companies(
        symbol=symbol, min_esg=min_esg, min_sharpe=min_sharpe, top_n=top_n,
    )
