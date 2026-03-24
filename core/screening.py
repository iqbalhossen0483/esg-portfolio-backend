"""ESG screening pipeline: hard screen → soft screen → select universe."""

import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Company, ComputedMetric


async def hard_screen(db: AsyncSession) -> list[str]:
    """Hard screening: exclude restricted business, severe controversy, ESG < 20.
    Returns list of eligible symbols.
    """
    result = await db.execute(
        select(ComputedMetric.symbol).join(
            Company, ComputedMetric.symbol == Company.symbol
        ).where(
            Company.restricted_business == False,
            Company.severe_controversy == False,
            ComputedMetric.avg_esg_composite >= 20,
            ComputedMetric.eligible_hard_screen == True,
        )
    )
    return [r[0] for r in result.all()]


async def soft_screen(
    db: AsyncSession,
    eligible_symbols: list[str],
    top_pct: float = 0.5,
) -> list[str]:
    """Soft screening: best-in-class within each sector (top 50% by composite score).
    Returns list of selected symbols.
    """
    result = await db.execute(
        select(
            ComputedMetric.symbol,
            Company.sector,
            ComputedMetric.composite_score,
        ).join(
            Company, ComputedMetric.symbol == Company.symbol
        ).where(
            ComputedMetric.symbol.in_(eligible_symbols),
            Company.sector.isnot(None),
        )
    )
    rows = result.all()

    if not rows:
        return eligible_symbols

    df = pd.DataFrame(rows, columns=["symbol", "sector", "composite_score"])
    df["rank_pct"] = df.groupby("sector")["composite_score"].rank(pct=True, method="average")
    selected = df[df["rank_pct"] >= top_pct]["symbol"].tolist()

    return selected


async def select_universe(
    db: AsyncSession,
    max_stocks: int = 30,
) -> list[str]:
    """Full screening pipeline: hard → soft → limit to max_stocks.
    Returns list of selected symbols for DRL training/inference.
    """
    eligible = await hard_screen(db)
    if not eligible:
        # Fallback: get top companies by composite score
        result = await db.execute(
            select(ComputedMetric.symbol)
            .order_by(ComputedMetric.composite_score.desc())
            .limit(max_stocks)
        )
        return [r[0] for r in result.all()]

    selected = await soft_screen(db, eligible)
    return selected[:max_stocks]
