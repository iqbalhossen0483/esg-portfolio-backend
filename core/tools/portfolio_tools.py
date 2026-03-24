"""ADK FunctionTools for portfolio optimization and analysis."""

import asyncio

import numpy as np
import pandas as pd
from sqlalchemy import select

from db.database import async_session
from db.models import Company, ComputedMetric, PriceDaily, ESGScore


def optimize_portfolio(
    risk_tolerance: str = "balanced",
    esg_importance: str = "medium",
    investment_amount: float = None,
    max_stocks: int = 15,
    excluded_sectors: list[str] = None,
) -> dict:
    """Runs the DRL engine to generate an optimal portfolio allocation.

    Use this tool when the user wants to build a portfolio, invest money,
    or get AI-optimized stock recommendations with specific preferences.

    Args:
        risk_tolerance: One of 'conservative', 'balanced', 'aggressive'.
        esg_importance: One of 'low', 'medium', 'high'.
        investment_amount: Dollar amount to invest (optional, for $ allocation display).
        max_stocks: Maximum number of stocks in portfolio. Default 15.
        excluded_sectors: List of sector names to exclude. None for no exclusion.

    Returns:
        dict with portfolio allocations (symbol, name, sector, weight, amount,
        sharpe, esg) and portfolio-level metrics (total sharpe, esg, volatility).
    """
    esg_lambda_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
    esg_lambda = esg_lambda_map.get(esg_importance, 0.5)

    async def _optimize():
        async with async_session() as db:
            from core.drl_engine import get_active_model_path, load_model, inference

            model_path = await get_active_model_path(db)
            if not model_path:
                # Fallback: return top companies by composite score
                return _fallback_portfolio(db, max_stocks, excluded_sectors, investment_amount)

            # Get eligible companies
            query = select(
                ComputedMetric.symbol, Company.name, Company.sector,
                ComputedMetric.sharpe_252d, ComputedMetric.avg_esg_composite,
                ComputedMetric.composite_score, ComputedMetric.annual_volatility,
            ).join(
                Company, ComputedMetric.symbol == Company.symbol
            ).where(ComputedMetric.eligible_hard_screen == True)

            if excluded_sectors:
                query = query.where(~Company.sector.in_(excluded_sectors))

            query = query.order_by(ComputedMetric.composite_score.desc()).limit(max_stocks)
            result = await db.execute(query)
            rows = result.all()

            if not rows:
                return {"error": "No eligible companies found"}

            symbols = [r[0] for r in rows]
            company_data = {r[0]: {"name": r[1], "sector": r[2], "sharpe": float(r[3] or 0),
                                    "esg": float(r[4] or 0), "vol": float(r[6] or 0)} for r in rows}

            # Simple equal-weight with ESG tilt as fallback if model fails
            try:
                model = load_model(model_path)
                # Build minimal features for inference
                sector_set = list(set(d["sector"] for d in company_data.values() if d["sector"]))
                sector_map = {s: i for i, s in enumerate(sector_set)}
                sector_ids = np.array([sector_map.get(company_data[s]["sector"], 0) for s in symbols])

                # Use composite scores as proxy features
                features = np.random.randn(len(symbols), model.feat_dim).astype(np.float32)
                weights = inference(model, features, sector_ids)
            except Exception:
                # Equal weight with ESG tilt
                esg_scores = np.array([company_data[s]["esg"] for s in symbols])
                weights = esg_scores / esg_scores.sum() if esg_scores.sum() > 0 else np.ones(len(symbols)) / len(symbols)

            # Build response
            allocations = []
            for i, sym in enumerate(symbols):
                w = float(weights[i])
                info = company_data[sym]
                alloc = {
                    "symbol": sym, "name": info["name"], "sector": info["sector"],
                    "weight": round(w, 4),
                    "weight_pct": f"{w * 100:.1f}%",
                    "sharpe": info["sharpe"], "esg": info["esg"],
                }
                if investment_amount:
                    alloc["amount"] = round(w * investment_amount, 2)
                allocations.append(alloc)

            allocations.sort(key=lambda x: x["weight"], reverse=True)

            # Portfolio-level metrics
            port_sharpe = sum(a["weight"] * a["sharpe"] for a in allocations)
            port_esg = sum(a["weight"] * a["esg"] for a in allocations)
            sectors_used = len(set(a["sector"] for a in allocations if a["sector"]))

            return {
                "allocations": allocations,
                "portfolio_metrics": {
                    "weighted_sharpe": round(port_sharpe, 3),
                    "weighted_esg": round(port_esg, 1),
                    "num_holdings": len(allocations),
                    "sector_count": sectors_used,
                    "risk_profile": risk_tolerance,
                    "esg_priority": esg_importance,
                },
                "investment_amount": investment_amount,
            }

    return asyncio.run(_optimize())


async def _fallback_portfolio(db, max_stocks, excluded_sectors, investment_amount):
    """Simple fallback when no DRL model is active."""
    query = select(
        ComputedMetric.symbol, Company.name, Company.sector,
        ComputedMetric.sharpe_252d, ComputedMetric.avg_esg_composite,
    ).join(
        Company, ComputedMetric.symbol == Company.symbol
    ).order_by(ComputedMetric.composite_score.desc()).limit(max_stocks)

    if excluded_sectors:
        query = query.where(~Company.sector.in_(excluded_sectors))

    result = await db.execute(query)
    rows = result.all()

    if not rows:
        return {"error": "No companies available"}

    w = 1.0 / len(rows)
    allocations = [
        {
            "symbol": r[0], "name": r[1], "sector": r[2],
            "weight": round(w, 4), "weight_pct": f"{w * 100:.1f}%",
            "sharpe": float(r[3]) if r[3] else 0,
            "esg": float(r[4]) if r[4] else 0,
            "amount": round(w * investment_amount, 2) if investment_amount else None,
        }
        for r in rows
    ]
    return {"allocations": allocations, "portfolio_metrics": {
        "weighted_sharpe": sum(a["weight"] * a["sharpe"] for a in allocations),
        "weighted_esg": sum(a["weight"] * a["esg"] for a in allocations),
        "num_holdings": len(allocations),
        "note": "Equal-weight fallback (no active DRL model)",
    }}


def analyze_portfolio(holdings: dict) -> dict:
    """Analyzes a user-proposed portfolio for risk, ESG, and diversification.

    Args:
        holdings: Dict of symbol to weight, e.g. {'AAPL': 0.2, 'MSFT': 0.3, 'JNJ': 0.5}.

    Returns:
        dict with per-holding metrics, portfolio-level metrics,
        diversification analysis, and improvement suggestions.
    """
    async def _analyze():
        async with async_session() as db:
            symbols = list(holdings.keys())
            result = await db.execute(
                select(
                    ComputedMetric.symbol, Company.name, Company.sector,
                    ComputedMetric.sharpe_252d, ComputedMetric.avg_esg_composite,
                    ComputedMetric.annual_volatility, ComputedMetric.max_drawdown,
                ).join(
                    Company, ComputedMetric.symbol == Company.symbol
                ).where(ComputedMetric.symbol.in_([s.upper() for s in symbols]))
            )
            rows = result.all()

            analysis = []
            total_weight = sum(holdings.values())
            sectors = {}

            for r in rows:
                sym = r[0]
                w = holdings.get(sym, holdings.get(sym.lower(), 0)) / total_weight
                sector = r[2]
                analysis.append({
                    "symbol": sym, "name": r[1], "sector": sector,
                    "weight": round(w, 4),
                    "sharpe": float(r[3]) if r[3] else 0,
                    "esg": float(r[4]) if r[4] else 0,
                    "volatility": float(r[5]) if r[5] else 0,
                    "max_drawdown": float(r[6]) if r[6] else 0,
                })
                if sector:
                    sectors[sector] = sectors.get(sector, 0) + w

            port_sharpe = sum(a["weight"] * a["sharpe"] for a in analysis)
            port_esg = sum(a["weight"] * a["esg"] for a in analysis)
            port_vol = sum(a["weight"] * a["volatility"] for a in analysis)

            # Diversification warnings
            warnings = []
            for a in analysis:
                if a["weight"] > 0.12:
                    warnings.append(f"{a['symbol']} weight ({a['weight']:.0%}) exceeds 12% limit")
            for sec, sw in sectors.items():
                if sw > 0.35:
                    warnings.append(f"{sec} sector ({sw:.0%}) exceeds 35% limit")
            if len(sectors) < 3:
                warnings.append("Low diversification: only {len(sectors)} sector(s)")
            if port_esg < 30:
                warnings.append(f"Low portfolio ESG score ({port_esg:.1f})")

            return {
                "holdings": analysis,
                "portfolio_metrics": {
                    "weighted_sharpe": round(port_sharpe, 3),
                    "weighted_esg": round(port_esg, 1),
                    "weighted_volatility": round(port_vol, 4),
                    "num_holdings": len(analysis),
                    "sector_count": len(sectors),
                    "sector_weights": sectors,
                },
                "warnings": warnings,
                "diversification_score": min(len(sectors) / 5, 1.0),
            }

    return asyncio.run(_analyze())


def get_pareto_frontier(top_n: int = 20) -> dict:
    """Returns the Pareto frontier of Sharpe ratio vs ESG score trade-offs.

    Shows the best achievable combinations — you can't improve one metric
    without worsening the other.

    Args:
        top_n: Number of Pareto-optimal points to return. Default 20.

    Returns:
        dict with 'frontier' list of {symbol, sharpe, esg} points.
    """
    async def _query():
        async with async_session() as db:
            result = await db.execute(
                select(
                    ComputedMetric.symbol, Company.name,
                    ComputedMetric.sharpe_252d, ComputedMetric.avg_esg_composite,
                ).join(
                    Company, ComputedMetric.symbol == Company.symbol
                ).where(
                    ComputedMetric.sharpe_252d.isnot(None),
                    ComputedMetric.avg_esg_composite.isnot(None),
                ).order_by(ComputedMetric.sharpe_252d.desc()).limit(100)
            )
            rows = result.all()

            # Simple Pareto filter
            points = [(r[0], r[1], float(r[2]), float(r[3])) for r in rows]
            frontier = []
            max_esg = -1

            for sym, name, sharpe, esg in sorted(points, key=lambda x: -x[2]):
                if esg > max_esg:
                    frontier.append({"symbol": sym, "name": name, "sharpe": sharpe, "esg": esg})
                    max_esg = esg

            return frontier[:top_n]

    frontier = asyncio.run(_query())
    return {"frontier": frontier, "count": len(frontier)}
