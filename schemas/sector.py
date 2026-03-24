from pydantic import BaseModel


class SectorRankingResponse(BaseModel):
    sector: str
    avg_sharpe: float
    avg_esg: float
    avg_volatility: float
    avg_return: float
    company_count: int
    composite_score: float


class SectorDetailResponse(BaseModel):
    sector: dict
    top_companies: list[dict]
