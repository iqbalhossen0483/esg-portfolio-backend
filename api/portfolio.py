from fastapi import APIRouter, Depends

from core.auth.dependencies import get_current_user
from core.response import success_response
from core.tools.portfolio_tools import (
    analyze_portfolio,
    get_pareto_frontier,
    optimize_portfolio,
)
from db.models import User
from schemas.portfolio import AnalyzeRequest, OptimizeRequest

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post("/optimize")
async def optimize(request: OptimizeRequest, user: User = Depends(get_current_user)):
    """Run DRL engine to generate optimal portfolio allocation."""
    data = await optimize_portfolio(
        risk_tolerance=request.risk_tolerance,
        esg_importance=request.esg_importance,
        investment_amount=request.investment_amount,
        max_stocks=request.max_stocks,
        excluded_sectors=request.excluded_sectors,
    )
    return success_response(data=data, message="Portfolio optimized successfully")


@router.post("/analyze")
async def analyze(request: AnalyzeRequest, user: User = Depends(get_current_user)):
    """Analyze a user-proposed portfolio."""
    data = await analyze_portfolio(holdings=request.holdings)
    return success_response(data=data, message="Portfolio analyzed successfully")


@router.get("/pareto")
async def pareto(top_n: int = 20, user: User = Depends(get_current_user)):
    """Get Sharpe vs ESG Pareto frontier."""
    data = await get_pareto_frontier(top_n=top_n)
    return success_response(data=data, message="Pareto frontier retrieved successfully")
