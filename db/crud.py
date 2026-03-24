import uuid
from datetime import datetime

from sqlalchemy import select, update, delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    User,
    RefreshToken,
    Company,
    PriceDaily,
    ESGScore,
    ComputedMetric,
    SectorRanking,
    PortfolioAllocation,
    PortfolioMetric,
    DRLModel,
    TrainingJob,
    KnowledgeBase,
)


# ═══════════════════════════════════════
# Users
# ═══════════════════════════════════════


async def create_user(db: AsyncSession, data: dict) -> User:
    user = User(**data)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> User | None:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


# ═══════════════════════════════════════
# Refresh Tokens
# ═══════════════════════════════════════


async def save_refresh_token(
    db: AsyncSession, user_id: uuid.UUID, token: str, expires_at: datetime
) -> RefreshToken:
    rt = RefreshToken(user_id=user_id, token=token, expires_at=expires_at)
    db.add(rt)
    await db.commit()
    return rt


async def get_refresh_token(db: AsyncSession, token: str) -> RefreshToken | None:
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token == token)
    )
    return result.scalar_one_or_none()


async def revoke_all_refresh_tokens(db: AsyncSession, user_id: uuid.UUID):
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.is_revoked == False)
        .values(is_revoked=True)
    )
    await db.commit()


# ═══════════════════════════════════════
# Companies
# ═══════════════════════════════════════


async def upsert_company(db: AsyncSession, data: dict):
    stmt = pg_insert(Company).values(**data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol"],
        set_={k: v for k, v in data.items() if k != "symbol"},
    )
    await db.execute(stmt)
    await db.commit()


async def get_company(db: AsyncSession, symbol: str) -> Company | None:
    result = await db.execute(select(Company).where(Company.symbol == symbol))
    return result.scalar_one_or_none()


async def list_companies(
    db: AsyncSession,
    sector: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Company]:
    query = select(Company)
    if sector:
        query = query.where(Company.sector == sector)
    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    return list(result.scalars().all())


# ═══════════════════════════════════════
# Prices
# ═══════════════════════════════════════


async def upsert_price(db: AsyncSession, data: dict):
    stmt = pg_insert(PriceDaily).values(**data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "date"],
        set_={k: v for k, v in data.items() if k not in ("symbol", "date")},
    )
    await db.execute(stmt)


async def bulk_upsert_prices(db: AsyncSession, records: list[dict]):
    if not records:
        return
    stmt = pg_insert(PriceDaily).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "date"],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
        },
    )
    await db.execute(stmt)
    await db.commit()


# ═══════════════════════════════════════
# ESG Scores
# ═══════════════════════════════════════


async def upsert_esg_score(db: AsyncSession, data: dict):
    stmt = pg_insert(ESGScore).values(**data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "date", "provider"],
        set_={
            k: v for k, v in data.items() if k not in ("symbol", "date", "provider")
        },
    )
    await db.execute(stmt)


async def bulk_upsert_esg_scores(db: AsyncSession, records: list[dict]):
    if not records:
        return
    stmt = pg_insert(ESGScore).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "date", "provider"],
        set_={
            "e_score": stmt.excluded.e_score,
            "s_score": stmt.excluded.s_score,
            "g_score": stmt.excluded.g_score,
            "composite_score": stmt.excluded.composite_score,
        },
    )
    await db.execute(stmt)
    await db.commit()


# ═══════════════════════════════════════
# Computed Metrics
# ═══════════════════════════════════════


async def upsert_computed_metric(db: AsyncSession, data: dict):
    stmt = pg_insert(ComputedMetric).values(**data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "as_of_date"],
        set_={k: v for k, v in data.items() if k not in ("symbol", "as_of_date")},
    )
    await db.execute(stmt)
    await db.commit()


async def get_computed_metrics(
    db: AsyncSession,
    sector: str | None = None,
    min_esg: float | None = None,
    min_sharpe: float | None = None,
    sort_by: str = "composite_score",
    limit: int = 50,
):
    query = select(ComputedMetric).join(
        Company, ComputedMetric.symbol == Company.symbol
    )
    if sector:
        query = query.where(Company.sector == sector)
    if min_esg is not None:
        query = query.where(ComputedMetric.avg_esg_composite >= min_esg)
    if min_sharpe is not None:
        query = query.where(ComputedMetric.sharpe_252d >= min_sharpe)

    sort_col = getattr(ComputedMetric, sort_by, ComputedMetric.composite_score)
    query = query.order_by(sort_col.desc()).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())


# ═══════════════════════════════════════
# Sector Rankings
# ═══════════════════════════════════════


async def upsert_sector_ranking(db: AsyncSession, data: dict):
    stmt = pg_insert(SectorRanking).values(**data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["sector", "as_of_date"],
        set_={k: v for k, v in data.items() if k not in ("sector", "as_of_date")},
    )
    await db.execute(stmt)
    await db.commit()


async def get_sector_rankings(
    db: AsyncSession, sort_by: str = "composite_score", limit: int = 20
):
    sort_col = getattr(SectorRanking, sort_by, SectorRanking.composite_score)
    result = await db.execute(
        select(SectorRanking).order_by(sort_col.desc()).limit(limit)
    )
    return list(result.scalars().all())


# ═══════════════════════════════════════
# DRL Models
# ═══════════════════════════════════════


async def create_drl_model(db: AsyncSession, data: dict) -> DRLModel:
    model = DRLModel(**data)
    db.add(model)
    await db.commit()
    await db.refresh(model)
    return model


async def get_active_model(db: AsyncSession) -> DRLModel | None:
    result = await db.execute(
        select(DRLModel).where(DRLModel.status == "active")
    )
    return result.scalar_one_or_none()


async def list_drl_models(db: AsyncSession) -> list[DRLModel]:
    result = await db.execute(
        select(DRLModel).order_by(DRLModel.created_at.desc())
    )
    return list(result.scalars().all())


async def activate_model(db: AsyncSession, model_id: uuid.UUID):
    # Retire all currently active models
    await db.execute(
        update(DRLModel).where(DRLModel.status == "active").values(status="retired")
    )
    # Activate the specified model
    await db.execute(
        update(DRLModel).where(DRLModel.model_id == model_id).values(status="active")
    )
    await db.commit()


# ═══════════════════════════════════════
# Training Jobs
# ═══════════════════════════════════════


async def create_training_job(db: AsyncSession, data: dict) -> TrainingJob:
    job = TrainingJob(**data)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def update_training_job(db: AsyncSession, job_id: uuid.UUID, data: dict):
    await db.execute(
        update(TrainingJob).where(TrainingJob.job_id == job_id).values(**data)
    )
    await db.commit()


async def get_training_job(db: AsyncSession, job_id: uuid.UUID) -> TrainingJob | None:
    result = await db.execute(
        select(TrainingJob).where(TrainingJob.job_id == job_id)
    )
    return result.scalar_one_or_none()


async def list_training_jobs(db: AsyncSession, limit: int = 20) -> list[TrainingJob]:
    result = await db.execute(
        select(TrainingJob).order_by(TrainingJob.started_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


# ═══════════════════════════════════════
# Knowledge Base
# ═══════════════════════════════════════


async def create_knowledge_entry(db: AsyncSession, data: dict) -> KnowledgeBase:
    entry = KnowledgeBase(**data)
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry


async def search_knowledge(db: AsyncSession, embedding, limit: int = 5):
    result = await db.execute(
        select(KnowledgeBase)
        .order_by(KnowledgeBase.embedding.cosine_distance(embedding))
        .limit(limit)
    )
    return list(result.scalars().all())
