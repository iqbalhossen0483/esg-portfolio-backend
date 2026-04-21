"""Celery task for DRL model training."""

import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import select

from .celery_app import celery_app
from db.database import async_session
from db.models import Company, PriceDaily, ESGScore
from db.crud import create_drl_model
from core.screening import select_universe
from drl.features import build_feature_tensor
from drl.train import train_mappo
from drl.evaluate import evaluate_model
from drl.models.mappo import MAPPO


async def _train_model(esg_lambda: float = 0.5, episodes: int = 500):
    """Full training pipeline: screen → features → train → evaluate → save."""
    async with async_session() as db:
        # Step 1: Select universe
        symbols = await select_universe(db, max_stocks=30)
        if len(symbols) < 5:
            return {"error": "Not enough eligible companies for training"}

        # Step 2: Load prices
        result = await db.execute(
            select(PriceDaily.symbol, PriceDaily.date, PriceDaily.close)
            .where(PriceDaily.symbol.in_(symbols))
            .order_by(PriceDaily.date)
        )
        rows = result.all()
        if not rows:
            return {"error": "No price data available"}

        df = pd.DataFrame(rows, columns=["symbol", "date", "close"])
        prices_df = df.pivot(index="date", columns="symbol", values="close")
        prices_df = prices_df.dropna(axis=1, thresh=int(len(prices_df) * 0.95))
        symbols = list(prices_df.columns)

        # Step 3: Load ESG scores
        esg_result = await db.execute(
            select(ESGScore.symbol, ESGScore.provider, ESGScore.composite_score)
            .where(ESGScore.symbol.in_(symbols))
        )
        esg_rows = esg_result.all()
        esg_dict = {}
        for sym, provider, score in esg_rows:
            if sym not in esg_dict:
                esg_dict[sym] = []
            if score is not None:
                esg_dict[sym].append(float(score))

        esg_scores_map = {}
        for sym in symbols:
            scores = esg_dict.get(sym, [50.0])
            avg = np.mean(scores) / 100.0  # normalize to 0-1
            esg_scores_map[sym] = (avg, avg)

        # Step 4: Get sector IDs
        company_result = await db.execute(
            select(Company.symbol, Company.sector).where(Company.symbol.in_(symbols))
        )
        company_rows = company_result.all()
        sector_map = {r[0]: r[1] for r in company_rows}

        unique_sectors = list(set(s for s in sector_map.values() if s))
        sector_to_id = {s: i for i, s in enumerate(unique_sectors)}
        sector_ids = np.array([
            sector_to_id.get(sector_map.get(sym, "Unknown"), 0) for sym in symbols
        ])
        esg_array = np.array([esg_scores_map.get(sym, (0.5, 0.5))[0] for sym in symbols])

        # Step 5: Build feature tensor
        features, returns, dates = build_feature_tensor(prices_df, esg_scores_map, symbols)

        # Step 6: Train
        train_result = train_mappo(
            features=features,
            returns=returns,
            sector_ids=sector_ids,
            esg_scores=esg_array,
            esg_lambda=esg_lambda,
            episodes=episodes,
        )

        # Step 7: Evaluate
        model = MAPPO.load(train_result["model_path"])
        eval_result = evaluate_model(
            model=model,
            features=features,
            returns=returns,
            sector_ids=sector_ids,
            esg_scores=esg_array,
        )

        # Step 8: Save to database
        model_record = await create_drl_model(db, {
            "model_name": f"mappo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_path": train_result["model_path"],
            "architecture": "mappo_ra_drl",
            "trained_at": datetime.now(timezone.utc),
            "train_sharpe": train_result.get("training_history", [{}])[-1].get("avg_reward", 0),
            "test_sharpe": eval_result.get("sharpe_ratio", 0),
            "train_esg": float(esg_array.mean() * 100),
            "test_esg": eval_result.get("avg_portfolio_esg", 0) * 100,
            "status": "trained",
            "hyperparameters": {
                "episodes": episodes,
                "esg_lambda": esg_lambda,
                "n_assets": len(symbols),
                "symbols": symbols,
            },
        })

        return {
            "model_id": model_record.id,
            "model_path": train_result["model_path"],
            "train_metrics": train_result,
            "eval_metrics": eval_result,
        }


@celery_app.task(name="tasks.train_drl_model")
def train_drl_model(esg_lambda: float = 0.5, episodes: int = 500):
    """Celery task: train a DRL model."""
    print(f"Training DRL model with ESG λ={esg_lambda}, episodes={episodes}")
    return asyncio.run(_train_model(esg_lambda, episodes))
