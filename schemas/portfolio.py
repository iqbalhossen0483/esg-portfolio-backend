from pydantic import BaseModel


class OptimizeRequest(BaseModel):
    risk_tolerance: str = "balanced"
    esg_importance: str = "medium"
    investment_amount: float | None = None
    max_stocks: int = 15
    excluded_sectors: list[str] | None = None


class AnalyzeRequest(BaseModel):
    holdings: dict[str, float]
