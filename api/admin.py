from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth.dependencies import require_admin
from core.auth.security import hash_password
from core.logging import get_logger
from core.response import success_response
from db.crud import (
    activate_model,
    create_user,
    get_user_by_email,
    list_drl_models,
)
from db.database import get_db
from db.models import User
from schemas.auth import ChangeUserRoleRequest, CreateUserRequest

log = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


class TrainModelRequest(BaseModel):
    esg_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    episodes: int = Field(default=500, ge=1, le=10_000)


# ═══════════════════════════════════════
# User Management (Admin only)
# ═══════════════════════════════════════


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user_by_admin(
    request: CreateUserRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new user with a specific role. Admin only.
    This is the only way to create admin accounts.
    """
    if request.role not in ("investor", "admin"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be 'investor' or 'admin'",
        )

    existing = await get_user_by_email(db, request.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = await create_user(
        db,
        {
            "email": request.email,
            "password_hash": hash_password(request.password),
            "full_name": request.full_name,
            "role": request.role,
            "is_verified": True,
        },
    )

    return success_response(
        data={
            "user_id": str(user.id),
            "email": user.email,
            "role": user.role,
        },
        message=f"{request.role.capitalize()} account created",
        status_code=201,
    )


@router.get("/users")
async def list_users(
    role: str | None = None,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all users. Optionally filter by role."""
    if role and role not in ("investor", "admin"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role filter must be 'investor' or 'admin'",
        )

    query = select(User)
    if role:
        query = query.where(User.role == role)
    query = query.order_by(User.created_at.desc())
    result = await db.execute(query)
    users = result.scalars().all()

    return success_response(
        data=[
            {
                "id": str(u.id),
                "email": u.email,
                "full_name": u.full_name,
                "role": u.role,
                "is_active": u.is_active,
                "is_verified": u.is_verified,
                "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
                "created_at": u.created_at.isoformat(),
            }
            for u in users
        ],
        message="Users retrieved successfully",
    )


@router.put("/users/{user_id}/role")
async def change_user_role(
    user_id: int,
    request: ChangeUserRoleRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Change a user's role. Admin only."""
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role",
        )

    await db.execute(
        update(User).where(User.id == user_id).values(role=request.role)
    )
    await db.commit()
    log.info("admin %s changed user %s role to %s", admin.id, user_id, request.role)
    return success_response(
        data={"user_id": str(user_id)},
        message=f"User role updated to {request.role}",
    )


@router.put("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: int,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Deactivate a user account. Admin only."""
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account",
        )

    await db.execute(
        update(User).where(User.id == user_id).values(is_active=False)
    )
    await db.commit()
    log.info("admin %s deactivated user %s", admin.id, user_id)
    return success_response(
        data={"user_id": str(user_id)},
        message="User deactivated",
    )


@router.put("/users/{user_id}/activate")
async def activate_user(
    user_id: int,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Reactivate a deactivated user. Admin only."""
    await db.execute(
        update(User).where(User.id == user_id).values(is_active=True)
    )
    await db.commit()
    return success_response(
        data={"user_id": str(user_id)},
        message="User activated",
    )


# ═══════════════════════════════════════
# Model Management (Admin only)
# ═══════════════════════════════════════


@router.post("/models/train")
async def trigger_training(
    request: TrainModelRequest,
    user: User = Depends(require_admin),
):
    """Trigger DRL model training via Celery."""
    from tasks.training_task import train_drl_model
    task = train_drl_model.delay(request.esg_lambda, request.episodes)
    return success_response(
        data={"status": "training_started", "task_id": task.id},
        message=f"Training started with {request.episodes} episodes, ESG λ={request.esg_lambda}",
    )


@router.get("/models")
async def get_models(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all trained DRL models with performance metrics."""
    models = await list_drl_models(db)
    return success_response(
        data=[
            {
                "model_id": m.id,
                "model_name": m.model_name,
                "architecture": m.architecture,
                "status": m.status,
                "trained_at": m.trained_at.isoformat() if m.trained_at else None,
                "train_sharpe": float(m.train_sharpe) if m.train_sharpe else None,
                "test_sharpe": float(m.test_sharpe) if m.test_sharpe else None,
                "train_esg": float(m.train_esg) if m.train_esg else None,
                "test_esg": float(m.test_esg) if m.test_esg else None,
                "hyperparameters": m.hyperparameters,
                "created_at": m.created_at.isoformat(),
            }
            for m in models
        ],
        message="Models retrieved successfully",
    )


@router.put("/models/{model_id}/activate")
async def activate_drl_model(
    model_id: int,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set a trained model as the active model for inference."""
    await activate_model(db, model_id)
    return success_response(
        data={"model_id": str(model_id)},
        message="Model activated successfully",
    )


@router.get("/pipeline/status")
async def get_pipeline_status(user: User = Depends(require_admin)):
    """Inspect Celery worker / queue activity for the ingestion + training pipelines."""
    from tasks.celery_app import celery_app

    try:
        inspector = celery_app.control.inspect(timeout=1.5)
        active = inspector.active() or {}
        scheduled = inspector.scheduled() or {}
        reserved = inspector.reserved() or {}
    except Exception:
        log.exception("celery inspect failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Celery broker unreachable",
        )

    def _flatten(by_worker: dict) -> list[dict]:
        return [
            {"worker": worker, **task}
            for worker, tasks in by_worker.items()
            for task in tasks
        ]

    return success_response(
        data={
            "workers": list(active.keys()),
            "active": _flatten(active),
            "scheduled": _flatten(scheduled),
            "reserved": _flatten(reserved),
        },
        message="Pipeline status retrieved",
    )
