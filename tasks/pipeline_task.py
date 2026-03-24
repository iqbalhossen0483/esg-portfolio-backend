import asyncio
from datetime import date

import pandas as pd
from sqlalchemy import select, func

from .celery_app import celery_app
from db.database import async_session
from db.models import Company, PriceDaily, ESGScore, ComputedMetric, SectorRanking
from db.crud import upsert_computed_metric, upsert_sector_ranking
from core.metrics import compute_all_metrics


async def _recompute_metrics():
    """Compute financial + ESG metrics for all companies and aggregate sector rankings."""
    async with async_session() as db:
        # Get all companies
        result = await db.execute(select(Company.symbol, Company.sector))
        companies = result.all()

        as_of = date.today()
        sector_data = {}

        for symbol, sector in companies:
            # Get prices
            price_result = await db.execute(
                select(PriceDaily.date, PriceDaily.close)
                .where(PriceDaily.symbol == symbol)
                .order_by(PriceDaily.date)
            )
            price_rows = price_result.all()
            if len(price_rows) < 60:
                continue

            prices = pd.Series(
                [float(r.close) for r in price_rows],
                index=[r.date for r in price_rows],
            )

            # Get latest ESG scores (average across providers)
            esg_result = await db.execute(
                select(
                    func.avg(ESGScore.composite_score).label("avg_composite"),
                    func.avg(ESGScore.e_score).label("avg_e"),
                    func.avg(ESGScore.s_score).label("avg_s"),
                    func.avg(ESGScore.g_score).label("avg_g"),
                ).where(ESGScore.symbol == symbol)
            )
            esg = esg_result.one_or_none()

            metrics = compute_all_metrics(
                symbol=symbol,
                prices=prices,
                esg_composite=float(esg.avg_composite) if esg and esg.avg_composite else None,
                e_score=float(esg.avg_e) if esg and esg.avg_e else None,
                s_score=float(esg.avg_s) if esg and esg.avg_s else None,
                g_score=float(esg.avg_g) if esg and esg.avg_g else None,
            )
            metrics["as_of_date"] = as_of
            await upsert_computed_metric(db, metrics)

            # Aggregate for sector rankings
            if sector:
                if sector not in sector_data:
                    sector_data[sector] = {
                        "sharpes": [],
                        "esgs": [],
                        "vols": [],
                        "returns": [],
                        "count": 0,
                    }
                sd = sector_data[sector]
                sd["count"] += 1
                if metrics.get("sharpe_252d") is not None:
                    sd["sharpes"].append(metrics["sharpe_252d"])
                if metrics.get("avg_esg_composite") is not None:
                    sd["esgs"].append(metrics["avg_esg_composite"])
                if metrics.get("annual_volatility") is not None:
                    sd["vols"].append(metrics["annual_volatility"])
                if metrics.get("annual_return") is not None:
                    sd["returns"].append(metrics["annual_return"])

        # Upsert sector rankings
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

    return {"status": "completed", "companies_processed": len(companies)}


@celery_app.task(name="tasks.recompute_metrics")
def recompute_metrics():
    """Celery task: recompute all financial + ESG metrics."""
    return asyncio.run(_recompute_metrics())
