import asyncio
from datetime import date

import pandas as pd
from sqlalchemy import select, func

from core.logging import get_logger
from core.metrics import compute_all_metrics
from db.crud import upsert_computed_metric, upsert_sector_ranking
from db.database import async_session
from db.models import Company, ESGScore, PriceDaily

from .celery_app import celery_app

log = get_logger(__name__)


async def _recompute_metrics():
    """Compute financial + ESG metrics for all companies and aggregate sector rankings.

    Bulk-loads prices and ESG aggregates in two queries instead of N+1.
    """
    async with async_session() as db:
        company_rows = (await db.execute(
            select(Company.symbol, Company.sector)
        )).all()
        sectors = {symbol: sector for symbol, sector in company_rows}

        # Bulk fetch all prices
        price_rows = (await db.execute(
            select(PriceDaily.symbol, PriceDaily.date, PriceDaily.close)
            .order_by(PriceDaily.symbol, PriceDaily.date)
        )).all()
        prices_by_symbol: dict[str, list[tuple]] = {}
        for symbol, dt, close in price_rows:
            if close is None:
                continue
            prices_by_symbol.setdefault(symbol, []).append((dt, float(close)))

        # Bulk fetch ESG aggregates per symbol
        esg_rows = (await db.execute(
            select(
                ESGScore.symbol,
                func.avg(ESGScore.composite_score).label("avg_composite"),
                func.avg(ESGScore.e_score).label("avg_e"),
                func.avg(ESGScore.s_score).label("avg_s"),
                func.avg(ESGScore.g_score).label("avg_g"),
            ).group_by(ESGScore.symbol)
        )).all()
        esg_by_symbol = {
            r.symbol: {
                "composite": float(r.avg_composite) if r.avg_composite is not None else None,
                "e": float(r.avg_e) if r.avg_e is not None else None,
                "s": float(r.avg_s) if r.avg_s is not None else None,
                "g": float(r.avg_g) if r.avg_g is not None else None,
            }
            for r in esg_rows
        }

        as_of = date.today()
        sector_data: dict[str, dict] = {}
        processed = 0

        for symbol, sector in company_rows:
            rows = prices_by_symbol.get(symbol)
            if not rows or len(rows) < 60:
                continue

            prices = pd.Series(
                [c for _, c in rows],
                index=[d for d, _ in rows],
            )

            esg = esg_by_symbol.get(symbol, {})
            metrics = compute_all_metrics(
                symbol=symbol,
                prices=prices,
                esg_composite=esg.get("composite"),
                e_score=esg.get("e"),
                s_score=esg.get("s"),
                g_score=esg.get("g"),
            )
            metrics["as_of_date"] = as_of
            await upsert_computed_metric(db, metrics)
            processed += 1

            if sector:
                sd = sector_data.setdefault(sector, {
                    "sharpes": [], "esgs": [], "vols": [], "returns": [], "count": 0,
                })
                sd["count"] += 1
                if metrics.get("sharpe_252d") is not None:
                    sd["sharpes"].append(metrics["sharpe_252d"])
                if metrics.get("avg_esg_composite") is not None:
                    sd["esgs"].append(metrics["avg_esg_composite"])
                if metrics.get("annual_volatility") is not None:
                    sd["vols"].append(metrics["annual_volatility"])
                if metrics.get("annual_return") is not None:
                    sd["returns"].append(metrics["annual_return"])

        for sector, sd in sector_data.items():
            avg_sharpe = sum(sd["sharpes"]) / len(sd["sharpes"]) if sd["sharpes"] else 0
            avg_esg = sum(sd["esgs"]) / len(sd["esgs"]) if sd["esgs"] else 0
            avg_vol = sum(sd["vols"]) / len(sd["vols"]) if sd["vols"] else 0
            avg_ret = sum(sd["returns"]) / len(sd["returns"]) if sd["returns"] else 0
            composite = 0.4 * avg_sharpe + 0.4 * (avg_esg / 100) + 0.2 * (1 - avg_vol)

            await upsert_sector_ranking(db, {
                "sector": sector,
                "as_of_date": as_of,
                "company_count": sd["count"],
                "avg_sharpe": avg_sharpe,
                "avg_esg": avg_esg,
                "avg_volatility": avg_vol,
                "avg_return": avg_ret,
                "composite_score": composite,
            })

    log.info("recompute done companies_processed=%d sectors=%d", processed, len(sector_data))
    return {"status": "completed", "companies_processed": processed}


@celery_app.task(name="tasks.recompute_metrics")
def recompute_metrics():
    """Celery task: recompute all financial + ESG metrics."""
    return asyncio.run(_recompute_metrics())
