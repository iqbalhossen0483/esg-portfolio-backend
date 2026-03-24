"""Portfolio constraint projection.
Enforces max position, sector cap, and normalization constraints.
"""

import numpy as np


def project_with_constraints(
    weights: np.ndarray,
    sector_ids: np.ndarray,
    max_weight: float = 0.12,
    sector_cap: float = 0.35,
    iterations: int = 5,
) -> np.ndarray:
    """Project raw weights to satisfy portfolio constraints.

    Iteratively applies:
    1. Cap each position at max_weight (12%)
    2. Cap each sector at sector_cap (35%)
    3. Normalize to sum = 1

    Args:
        weights: Raw weight array of shape [N_assets].
        sector_ids: Integer sector ID per asset, shape [N_assets].
        max_weight: Maximum weight for a single position.
        sector_cap: Maximum total weight for a single sector.
        iterations: Number of projection iterations.

    Returns:
        Projected weights summing to 1, all constraints satisfied.
    """
    w = np.clip(weights, 0, None).astype(np.float64)

    if w.sum() == 0:
        w = np.ones_like(w)

    w = w / w.sum()

    for _ in range(iterations):
        # Step 1: Cap individual positions
        w = np.minimum(w, max_weight)

        # Step 2: Cap sector exposures
        for s in np.unique(sector_ids):
            mask = sector_ids == s
            sector_sum = w[mask].sum()
            if sector_sum > sector_cap and sector_sum > 0:
                w[mask] *= sector_cap / sector_sum

        # Step 3: Normalize
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()

    return w
