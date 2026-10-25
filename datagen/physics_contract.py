"""Shared, NumPy-only contract for the existing two-group model equations.

Removal includes absorption, outgoing scatter and any axial leakage already
used by the reference solver. Fission spectrum is [1, 0], with no upscatter.
"""

from __future__ import annotations

import numpy as np

XS_FIELDS = ("D1", "D2", "sig_r1", "sig_r2", "nu_sig_f1", "nu_sig_f2", "sig_s12")
N_XS = len(XS_FIELDS)
MARSHAK = 0.5
QUARTER_CORE_BC = {"west": "reflective", "south": "reflective", "east": "vacuum", "north": "vacuum"}


def validate_xs(xs: np.ndarray, mask: np.ndarray) -> None:
    xs, mask = np.asarray(xs), np.asarray(mask, dtype=bool)
    if xs.shape != (N_XS, *mask.shape) or mask.ndim != 2 or not mask.any():
        raise ValueError("Expected seven cross-section maps and a nonempty 2-D domain mask")
    values = xs[:, mask]
    if not np.isfinite(xs).all() or (values[:2] <= 0).any() or (values[2:] < 0).any():
        raise ValueError("Active D must be positive; all other XS must be nonnegative and finite")
    if (values[2] + 1e-12 < values[6]).any():
        raise ValueError("Fast removal must include outgoing fast-to-thermal scatter")
    if not (values[4:6] > 0).any():
        raise ValueError("An eigenvalue configuration must contain fissile material")
    if np.any(xs[:, ~mask] != 0):
        raise ValueError("XS outside the physical domain must be zero padding")


def resolve_xs(material_map, xs_table, rod_state=None, rodded_xs_table=None, mask=None):
    """Resolve configurable layout, base properties and rod endpoints to [7,H,W].

    Fractional rod state is an effective linear XS interpolation, not a
    calibrated axial insertion depth. There is no assumed universal rod worth.
    """
    ids = np.asarray(material_map)
    if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("material_map must be a 2-D integer array")
    mask = ids >= 0 if mask is None else np.asarray(mask, bool)
    if mask.shape != ids.shape or not mask.any():
        raise ValueError("Invalid physical domain mask")
    base = np.asarray(xs_table, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != N_XS or not np.isfinite(base).all():
        raise ValueError(f"xs_table columns must be {XS_FIELDS}")
    if ids[mask].min() < 0 or ids[mask].max() >= len(base):
        raise ValueError("Active material ID is outside the supplied property table")
    rod = np.zeros(ids.shape) if rod_state is None else np.asarray(rod_state, dtype=float)
    if rod.shape != ids.shape or not np.isfinite(rod).all() or ((rod < 0) | (rod > 1)).any():
        raise ValueError("rod_state must be a finite [H,W] map in [0,1]")
    if np.any(rod[~mask] != 0):
        raise ValueError("Rods cannot be placed outside the physical domain")
    active = base[ids[mask]].copy()
    if np.any(rod[mask] != 0):
        if rodded_xs_table is None:
            raise ValueError("Nonzero rods require explicit rodded_xs_table endpoints")
        rodded = np.asarray(rodded_xs_table, dtype=float)
        if rodded.shape != base.shape or not np.isfinite(rodded).all():
            raise ValueError("Rodded and base property tables must have the same shape")
        active += rod[mask, None] * (rodded[ids[mask]] - active)
    xs = np.zeros((N_XS, *ids.shape), dtype=np.float64)
    xs[:, mask] = active.T
    validate_xs(xs, mask)
    return xs


def compact_xs(xs, mask):
    """Build per-case physics material IDs and a table, preserving exact fields.

    Unlike semantic layout IDs, these IDs distinguish changes in continuous
    properties and partial rod strengths. The final table row is void.
    """
    mask = np.asarray(mask, bool)
    validate_xs(xs, mask)
    table, inverse = np.unique(np.asarray(xs)[:, mask].T, axis=0, return_inverse=True)
    void_id = len(table)
    ids = np.full(mask.shape, void_id, dtype=np.int64)
    ids[mask] = inverse
    return ids, np.vstack([table, np.zeros((1, N_XS))]), void_id
