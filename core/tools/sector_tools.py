"""ADK FunctionTools for sector analysis. Async — safe for FastAPI routes
and ADK FunctionTool wrapping."""

from sqlalchemy import select

from db.crud import get_sector_rankings as _get_rankings
from db.database import async_session
from db.models import Company, ComputedMetric, SectorRanking


_SORT_MAP = {
    "composite": "composite_score",
    "sharpe": "avg_sharpe",
    "esg": "avg_esg",
    "low_risk": "avg_volatility",
}


async def get_sector_rankings(sort_by: str = "composite", top_n: int = 5) -> dict:
    """Returns top market sectors ranked by the specified metric.

    Use this tool when the user asks about which sectors or industries
    are best for investing, ESG performance, or risk-adjusted returns.

    Args:
        sort_by: Ranking criteria. Options:
            - 'composite': Combined score of Sharpe + ESG + low risk (default)
            - 'sharpe': Best risk-adjusted returns (avg_sharpe)
            - 'esg': Highest ESG scores (avg_esg)
            - 'low_risk': Lowest volatility (avg_volatility ascending)
        top_n: Number of top sectors to return. Default 5, max 11.

    Returns:
        dict with 'sectors' list containing objects with fields:
        sector, avg_sharpe, avg_esg, avg_volatility, avg_return,
        company_count, composite_score.
    """
    sort_col = _SORT_MAP.get(sort_by, "composite_score")
    limit = max(1, min(top_n, 11))

    async with async_session() as db:
        rankings = await _get_rankings(db, sort_by=sort_col, limit=limit)
        sectors = [
            {
                "sector": r.sector,
                "avg_sharpe": float(r.avg_sharpe) if r.avg_sharpe else 0,
                "avg_esg": float(r.avg_esg) if r.avg_esg else 0,
                "avg_volatility": float(r.avg_volatility) if r.avg_volatility else 0,
                "avg_return": float(r.avg_return) if r.avg_return else 0,
                "company_count": r.company_count or 0,
                "composite_score": float(r.composite_score) if r.composite_score else 0,
            }
            for r in rankings
        ]

    return {"sectors": sectors, "count": len(sectors), "sorted_by": sort_by}


async def get_sector_detail(sector: str) -> dict:
    """Returns detailed information about a specific sector including its top companies.

    Args:
        sector: Sector name (e.g., 'Technology', 'Healthcare', 'Financials').

    Returns:
        dict with sector metrics and top 10 companies in that sector
        with their individual Sharpe, ESG, volatility, and composite scores.
    """
    async with async_session() as db:
        sr = await db.execute(
            select(SectorRanking).where(SectorRanking.sector == sector).limit(1)
        )
        ranking = sr.scalar_one_or_none()

        result = await db.execute(
            select(
                ComputedMetric.symbol,
                Company.name,
                ComputedMetric.sharpe_252d,
                ComputedMetric.avg_esg_composite,
                ComputedMetric.annual_volatility,
                ComputedMetric.annual_return,
                ComputedMetric.composite_score,
            ).join(
                Company, ComputedMetric.symbol == Company.symbol
            ).where(
                Company.sector == sector
            ).order_by(
                ComputedMetric.composite_score.desc()
            ).limit(10)
        )
        companies = [
            {
                "symbol": r[0],
                "name": r[1],
                "sharpe": float(r[2]) if r[2] else 0,
                "esg": float(r[3]) if r[3] else 0,
                "volatility": float(r[4]) if r[4] else 0,
                "annual_return": float(r[5]) if r[5] else 0,
                "composite_score": float(r[6]) if r[6] else 0,
            }
            for r in result.all()
        ]

        sector_info = {}
        if ranking:
            sector_info = {
                "sector": ranking.sector,
                "avg_sharpe": float(ranking.avg_sharpe) if ranking.avg_sharpe else 0,
                "avg_esg": float(ranking.avg_esg) if ranking.avg_esg else 0,
                "avg_volatility": float(ranking.avg_volatility) if ranking.avg_volatility else 0,
                "company_count": ranking.company_count or 0,
            }

        return {"sector": sector_info, "top_companies": companies}
