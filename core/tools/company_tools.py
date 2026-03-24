"""ADK FunctionTools for company analysis."""

import asyncio

from sqlalchemy import select

from db.database import async_session
from db.models import Company, ComputedMetric


def get_best_companies(
    sector: str = None,
    min_esg: float = None,
    min_sharpe: float = None,
    top_n: int = 10,
) -> dict:
    """Returns best companies ranked by composite score with optional filters.

    Use this tool when the user asks about the best companies to invest in,
    optionally filtered by sector, minimum ESG score, or minimum Sharpe ratio.

    Args:
        sector: Filter by sector name (e.g., 'Technology'). None for all sectors.
        min_esg: Minimum ESG composite score (0-100). None for no filter.
        min_sharpe: Minimum Sharpe ratio. None for no filter.
        top_n: Number of companies to return. Default 10.

    Returns:
        dict with 'companies' list containing symbol, name, sector, sharpe,
        esg, volatility, annual_return, max_drawdown, composite_score.
    """
    async def _query():
        async with async_session() as db:
            query = select(
                ComputedMetric.symbol,
                Company.name,
                Company.sector,
                ComputedMetric.sharpe_252d,
                ComputedMetric.avg_esg_composite,
                ComputedMetric.annual_volatility,
                ComputedMetric.annual_return,
                ComputedMetric.max_drawdown,
                ComputedMetric.composite_score,
            ).join(Company, ComputedMetric.symbol == Company.symbol)

            if sector:
                query = query.where(Company.sector == sector)
            if min_esg is not None:
                query = query.where(ComputedMetric.avg_esg_composite >= min_esg)
            if min_sharpe is not None:
                query = query.where(ComputedMetric.sharpe_252d >= min_sharpe)

            query = query.order_by(ComputedMetric.composite_score.desc()).limit(top_n)
            result = await db.execute(query)

            return [
                {
                    "symbol": r[0], "name": r[1], "sector": r[2],
                    "sharpe": float(r[3]) if r[3] else 0,
                    "esg": float(r[4]) if r[4] else 0,
                    "volatility": float(r[5]) if r[5] else 0,
                    "annual_return": float(r[6]) if r[6] else 0,
                    "max_drawdown": float(r[7]) if r[7] else 0,
                    "composite_score": float(r[8]) if r[8] else 0,
                }
                for r in result.all()
            ]

    companies = asyncio.run(_query())
    return {"companies": companies, "count": len(companies), "filters": {
        "sector": sector, "min_esg": min_esg, "min_sharpe": min_sharpe,
    }}


def get_company_detail(symbol: str) -> dict:
    """Returns comprehensive detail for a single company including all metrics and ESG breakdown.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT').

    Returns:
        dict with full company profile: name, sector, sharpe, sortino, calmar,
        annual_return, volatility, max_drawdown, momentum, ESG breakdown (E/S/G),
        sector rank, eligibility status.
    """
    async def _query():
        async with async_session() as db:
            result = await db.execute(
                select(Company, ComputedMetric).join(
                    ComputedMetric, Company.symbol == ComputedMetric.symbol
                ).where(Company.symbol == symbol.upper())
            )
            row = result.first()
            if not row:
                return {"error": f"Company {symbol} not found"}

            c, m = row
            return {
                "symbol": c.symbol, "name": c.name, "sector": c.sector,
                "sub_industry": c.sub_industry,
                "restricted_business": c.restricted_business,
                "severe_controversy": c.severe_controversy,
                "sharpe": float(m.sharpe_252d) if m.sharpe_252d else None,
                "sortino": float(m.sortino_252d) if m.sortino_252d else None,
                "calmar": float(m.calmar_ratio) if m.calmar_ratio else None,
                "annual_return": float(m.annual_return) if m.annual_return else None,
                "volatility": float(m.annual_volatility) if m.annual_volatility else None,
                "max_drawdown": float(m.max_drawdown) if m.max_drawdown else None,
                "momentum_20d": float(m.momentum_20d) if m.momentum_20d else None,
                "momentum_60d": float(m.momentum_60d) if m.momentum_60d else None,
                "esg_composite": float(m.avg_esg_composite) if m.avg_esg_composite else None,
                "e_score": float(m.avg_e_score) if m.avg_e_score else None,
                "s_score": float(m.avg_s_score) if m.avg_s_score else None,
                "g_score": float(m.avg_g_score) if m.avg_g_score else None,
                "sector_rank_pct": float(m.sector_rank_pct) if m.sector_rank_pct else None,
                "eligible": m.eligible_hard_screen,
                "composite_score": float(m.composite_score) if m.composite_score else None,
            }

    return asyncio.run(_query())


def compare_companies(symbols: list[str]) -> dict:
    """Compares two or more companies on all financial and ESG metrics side-by-side.

    Args:
        symbols: List of ticker symbols to compare (e.g., ['AAPL', 'MSFT', 'GOOGL']).

    Returns:
        dict with 'companies' list, each containing full metrics for comparison.
    """
    results = []
    for sym in symbols:
        detail = get_company_detail(sym)
        if "error" not in detail:
            results.append(detail)

    return {"companies": results, "count": len(results)}


def search_similar_companies(
    symbol: str,
    min_esg: float = None,
    min_sharpe: float = None,
    top_n: int = 5,
) -> dict:
    """Finds companies with similar profiles using pgvector semantic similarity.

    Supports hybrid SQL + vector query: vector similarity for profile matching,
    SQL filters for ESG/Sharpe thresholds.

    Args:
        symbol: Reference company ticker (e.g., 'AAPL').
        min_esg: Minimum ESG score filter. None for no filter.
        min_sharpe: Minimum Sharpe ratio filter. None for no filter.
        top_n: Number of similar companies to return. Default 5.

    Returns:
        dict with 'similar_companies' list ranked by similarity.
    """
    async def _query():
        async with async_session() as db:
            # Get reference company embedding
            ref = await db.execute(
                select(Company.profile_embedding).where(Company.symbol == symbol.upper())
            )
            ref_row = ref.scalar_one_or_none()
            if ref_row is None:
                return {"error": f"Company {symbol} not found or has no embedding"}

            # Hybrid query: vector similarity + SQL filters
            query = select(
                Company.symbol, Company.name, Company.sector,
                ComputedMetric.sharpe_252d, ComputedMetric.avg_esg_composite,
                ComputedMetric.composite_score,
                Company.profile_embedding.cosine_distance(ref_row).label("distance"),
            ).join(
                ComputedMetric, Company.symbol == ComputedMetric.symbol
            ).where(
                Company.symbol != symbol.upper(),
                Company.profile_embedding.isnot(None),
            )

            if min_esg is not None:
                query = query.where(ComputedMetric.avg_esg_composite >= min_esg)
            if min_sharpe is not None:
                query = query.where(ComputedMetric.sharpe_252d >= min_sharpe)

            query = query.order_by("distance").limit(top_n)
            result = await db.execute(query)

            return [
                {
                    "symbol": r[0], "name": r[1], "sector": r[2],
                    "sharpe": float(r[3]) if r[3] else 0,
                    "esg": float(r[4]) if r[4] else 0,
                    "composite_score": float(r[5]) if r[5] else 0,
                    "similarity": round(1 - float(r[6]), 3) if r[6] else 0,
                }
                for r in result.all()
            ]

    similar = asyncio.run(_query())
    return {"reference": symbol, "similar_companies": similar, "count": len(similar)}
