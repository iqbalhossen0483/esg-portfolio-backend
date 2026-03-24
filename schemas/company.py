from pydantic import BaseModel


class CompanyListItem(BaseModel):
    symbol: str
    name: str
    sector: str | None
    sharpe: float | None
    esg: float | None
    volatility: float | None
    annual_return: float | None
    composite_score: float | None


class CompanyDetail(BaseModel):
    symbol: str
    name: str
    sector: str | None
    sub_industry: str | None
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    annual_return: float | None
    volatility: float | None
    max_drawdown: float | None
    momentum_20d: float | None
    momentum_60d: float | None
    esg_composite: float | None
    e_score: float | None
    s_score: float | None
    g_score: float | None
    sector_rank_pct: float | None
    eligible: bool | None
    composite_score: float | None
