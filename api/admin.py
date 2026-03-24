import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth.dependencies import require_admin
from core.auth.security import hash_password
from db.crud import (
    activate_model,
    create_user,
    get_user_by_email,
    list_drl_models,
)
from db.database import get_db
from db.models import User
from schemas.auth import CreateUserRequest

router = APIRouter(prefix="/admin", tags=["admin"])


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

    return {
        "message": f"{request.role.capitalize()} account created",
        "user_id": str(user.id),
        "email": user.email,
        "role": user.role,
    }


@router.get("/users")
async def list_users(
    role: str = None,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all users. Optionally filter by role."""
    query = select(User)
    if role:
        query = query.where(User.role == role)
    query = query.order_by(User.created_at.desc())
    result = await db.execute(query)
    users = result.scalars().all()

    return [
        {
            "id": str(u.id),
            "email": u.email,
            "full_name": u.full_name,
            "role": u.role,
            "is_active": u.is_active,
            "is_verified": u.is_verified,
            "last_login_at": str(u.last_login_at) if u.last_login_at else None,
            "created_at": str(u.created_at),
        }
        for u in users
    ]


@router.put("/users/{user_id}/role")
async def change_user_role(
    user_id: str,
    role: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Change a user's role. Admin only."""
    if role not in ("investor", "admin"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be 'investor' or 'admin'",
        )

    target_id = uuid.UUID(user_id)
    if target_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role",
        )

    await db.execute(
        update(User).where(User.id == target_id).values(role=role)
    )
    await db.commit()
    return {"message": f"User role updated to {role}", "user_id": user_id}


@router.put("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Deactivate a user account. Admin only."""
    target_id = uuid.UUID(user_id)
    if target_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account",
        )

    await db.execute(
        update(User).where(User.id == target_id).values(is_active=False)
    )
    await db.commit()
    return {"message": "User deactivated", "user_id": user_id}


@router.put("/users/{user_id}/activate")
async def activate_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Reactivate a deactivated user. Admin only."""
    await db.execute(
        update(User).where(User.id == uuid.UUID(user_id)).values(is_active=True)
    )
    await db.commit()
    return {"message": "User activated", "user_id": user_id}


# ═══════════════════════════════════════
# Model Management (Admin only)
# ═══════════════════════════════════════


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
    return {"status": "idle", "message": "No active jobs"}
