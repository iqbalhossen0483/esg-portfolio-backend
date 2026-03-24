import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth.dependencies import require_admin
from db.crud import activate_model, get_active_model, list_drl_models
from db.database import get_db
from db.models import User

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/models/train")
async def trigger_training(
    esg_lambda: float = 0.5,
    episodes: int = 500,
    user: User = Depends(require_admin),
):
    """Trigger DRL model training via Celery."""
    from tasks.training_task import train_drl_model
    task = train_drl_model.delay(esg_lambda, episodes)
    return {
        "status": "training_started",
        "task_id": task.id,
        "message": f"Training started with {episodes} episodes, ESG λ={esg_lambda}",
    }


@router.get("/models")
async def get_models(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all trained DRL models with performance metrics."""
    models = await list_drl_models(db)
    return [
        {
            "model_id": str(m.model_id),
            "model_name": m.model_name,
            "architecture": m.architecture,
            "status": m.status,
            "trained_at": str(m.trained_at) if m.trained_at else None,
            "train_sharpe": float(m.train_sharpe) if m.train_sharpe else None,
            "test_sharpe": float(m.test_sharpe) if m.test_sharpe else None,
            "train_esg": float(m.train_esg) if m.train_esg else None,
            "test_esg": float(m.test_esg) if m.test_esg else None,
            "hyperparameters": m.hyperparameters,
            "created_at": str(m.created_at),
        }
        for m in models
    ]


@router.put("/models/{model_id}/activate")
async def activate_drl_model(
    model_id: str,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set a trained model as the active model for inference."""
    await activate_model(db, uuid.UUID(model_id))
    return {"status": "activated", "model_id": model_id}


@router.get("/pipeline/status")
async def get_pipeline_status(user: User = Depends(require_admin)):
    """Check current pipeline/training job status."""
    # TODO: Query Celery for active tasks
    return {"status": "idle", "message": "No active jobs"}
