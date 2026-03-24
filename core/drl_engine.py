"""DRL inference engine. Loads trained model and runs fast forward pass."""

import numpy as np
import torch

from drl.models.mappo import MAPPO
from drl.constraints import project_with_constraints

# Cached model in memory
_active_model: MAPPO | None = None
_active_model_path: str | None = None


def load_model(model_path: str) -> MAPPO:
    """Load a MAPPO model from checkpoint."""
    global _active_model, _active_model_path
    if _active_model_path == model_path and _active_model is not None:
        return _active_model
    _active_model = MAPPO.load(model_path)
    _active_model_path = model_path
    return _active_model


def inference(
    model: MAPPO,
    features: np.ndarray,
    sector_ids: np.ndarray,
    max_weight: float = 0.12,
    sector_cap: float = 0.35,
) -> np.ndarray:
    """Run DRL inference to get portfolio weights.

    Args:
        model: Loaded MAPPO model.
        features: Current state [N_assets, F].
        sector_ids: Shape [N_assets].
        max_weight: Max single position weight.
        sector_cap: Max sector weight.

    Returns:
        Constrained portfolio weights [N_assets].
    """
    obs = torch.tensor(features, dtype=torch.float32)
    raw_weights = model.get_weights(obs).numpy()
    return project_with_constraints(raw_weights, sector_ids, max_weight, sector_cap)


async def get_active_model_path(db) -> str | None:
    """Get the file path of the currently active DRL model."""
    from db.crud import get_active_model
    model = await get_active_model(db)
    if model and model.model_path:
        return model.model_path
    return None
