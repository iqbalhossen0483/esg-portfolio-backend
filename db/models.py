from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    BigInteger,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ═══════════════════════════════════════════════════════════════
# Authentication & Users
# ═══════════════════════════════════════════════════════════════


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="investor")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    avatar_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default="now()")
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default="now()", onupdate=datetime.utcnow)

    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default="now()")

    user: Mapped["User"] = relationship(back_populates="refresh_tokens")


Index("idx_users_email", User.email)
Index("idx_refresh_tokens_user", RefreshToken.user_id)
Index("idx_refresh_tokens_token", RefreshToken.token)


# ═══════════════════════════════════════════════════════════════
# Business Data
# ═══════════════════════════════════════════════════════════════


class Company(Base):
    __tablename__ = "companies"

    symbol: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    sub_industry: Mapped[str | None] = mapped_column(String(200), nullable=True)
    market_cap: Mapped[float | None] = mapped_column(Numeric(18, 2), nullable=True)
    restricted_business: Mapped[bool] = mapped_column(Boolean, default=False)
    severe_controversy: Mapped[bool] = mapped_column(Boolean, default=False)
    profile_embedding = mapped_column(Vector(768), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default="now()")
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default="now()", onupdate=datetime.utcnow)


class PriceDaily(Base):
    __tablename__ = "prices_daily"

    symbol: Mapped[str] = mapped_column(
        String(10), ForeignKey("companies.symbol"), primary_key=True
    )
    date: Mapped[datetime] = mapped_column(Date, primary_key=True)
    open: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    high: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    low: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    close: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class ESGScore(Base):
    __tablename__ = "esg_scores"

    symbol: Mapped[str] = mapped_column(
        String(10), ForeignKey("companies.symbol"), primary_key=True
    )
    date: Mapped[datetime] = mapped_column(Date, primary_key=True)
    provider: Mapped[str] = mapped_column(String(50), primary_key=True)
    e_score: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    s_score: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    g_score: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    composite_score: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)


class ComputedMetric(Base):
    __tablename__ = "computed_metrics"

    symbol: Mapped[str] = mapped_column(
        String(10), ForeignKey("companies.symbol"), primary_key=True
    )
    as_of_date: Mapped[datetime] = mapped_column(Date, primary_key=True)
    annual_return: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    annual_volatility: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    sharpe_252d: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    sortino_252d: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    calmar_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    momentum_20d: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    momentum_60d: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    avg_esg_composite: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    avg_e_score: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    avg_s_score: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    avg_g_score: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    eligible_hard_screen: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    sector_rank_pct: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    composite_score: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)


class SectorRanking(Base):
    __tablename__ = "sector_rankings"

    sector: Mapped[str] = mapped_column(String(100), primary_key=True)
    as_of_date: Mapped[datetime] = mapped_column(Date, primary_key=True)
    company_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    avg_esg: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    avg_volatility: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    avg_return: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    composite_score: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)


class PortfolioAllocation(Base):
    __tablename__ = "portfolio_allocations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_profile: Mapped[str] = mapped_column(String(20), nullable=False)
    esg_priority: Mapped[str] = mapped_column(String(20), nullable=False)
    symbol: Mapped[str] = mapped_column(String(10), ForeignKey("companies.symbol"), nullable=False)
    weight: Mapped[float | None] = mapped_column(Numeric(6, 4), nullable=True)
    as_of_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    model_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("drl_models.id"), nullable=True)


class PortfolioMetric(Base):
    __tablename__ = "portfolio_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    risk_profile: Mapped[str | None] = mapped_column(String(20), nullable=True)
    esg_priority: Mapped[str | None] = mapped_column(String(20), nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    sortino_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    annual_return: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    annual_volatility: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    avg_esg_score: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    num_holdings: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sector_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    as_of_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default="now()")


class DRLModel(Base):
    __tablename__ = "drl_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    architecture: Mapped[str | None] = mapped_column(String(50), nullable=True)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    train_sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    test_sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    train_esg: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    test_esg: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="training")
    hyperparameters: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default="now()")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="processing")
    total_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunks_processed: Mapped[int] = mapped_column(Integer, default=0)
    records_stored: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    quality_report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default="now()")
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ═══════════════════════════════════════════════════════════════
# Vector Search (pgvector)
# ═══════════════════════════════════════════════════════════════


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    topic: Mapped[str | None] = mapped_column(String(50), nullable=True)
    embedding = mapped_column(Vector(768), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default="now()")


# pgvector indexes
Index(
    "idx_company_profile_embedding",
    Company.profile_embedding,
    postgresql_using="ivfflat",
    postgresql_with={"lists": 20},
    postgresql_ops={"profile_embedding": "vector_cosine_ops"},
)

Index(
    "idx_knowledge_base_embedding",
    KnowledgeBase.embedding,
    postgresql_using="ivfflat",
    postgresql_with={"lists": 10},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)
