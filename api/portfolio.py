from fastapi import APIRouter, Depends

from core.auth.dependencies import get_current_user
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
    return optimize_portfolio(
        risk_tolerance=request.risk_tolerance,
        esg_importance=request.esg_importance,
        investment_amount=request.investment_amount,
        max_stocks=request.max_stocks,
        excluded_sectors=request.excluded_sectors,
    )


@router.post("/analyze")
async def analyze(request: AnalyzeRequest, user: User = Depends(get_current_user)):
    """Analyze a user-proposed portfolio."""
    return analyze_portfolio(holdings=request.holdings)


@router.get("/pareto")
async def pareto(top_n: int = 20, user: User = Depends(get_current_user)):
    """Get Sharpe vs ESG Pareto frontier."""
    return get_pareto_frontier(top_n=top_n)
