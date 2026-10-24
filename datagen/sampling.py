"""Sample core configurations from the design space defined in a config file."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def sample_continuous(bounds: dict[str, tuple[float, float]], n: int, seed: int) -> list[dict]:
    """Latin hypercube samples over continuous parameters (enrichment, boron, ...)."""
    names = list(bounds)
    lo = np.array([bounds[k][0] for k in names])
    hi = np.array([bounds[k][1] for k in names])
    unit = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    pts = qmc.scale(unit, lo, hi)
    return [dict(zip(names, row)) for row in pts]


def sample_layout(grid_shape: tuple[int, int], n_fuel_types: int, rng: np.random.Generator) -> np.ndarray:
    """Random quarter-core assembly layout. Returns [H, W] int map.

    TODO: enforce core boundary shape (reflector cells), valid loading patterns.
    """
    raise NotImplementedError
