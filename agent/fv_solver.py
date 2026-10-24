"""Reference 2-D, 2-group finite-volume diffusion eigen-solver (stand-in backend for the agent).

Same discretization as `model.physics.fv_residual` (cell-centered FV, harmonic-mean interface D,
vacuum faces D dphi/dn + gamma phi = 0, reflective faces zero-current), so a solution on the
assembly grid has ~zero `fv_residual`. Each assembly is split into `n_sub x n_sub` mesh cells;
results are averaged back to the assembly grid.

This is NOT the training-data solver (that is ndiffusion, see datagen/). It exists so the agent
can answer questions before the surrogate is trained. Units: cm, cm^-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigs, splu

MARSHAK = 0.5
QUARTER_CORE_BC = {"west": "reflective", "south": "reflective", "east": "vacuum", "north": "vacuum"}


@dataclass
class FVSolution:
    k_eff: float
    flux: np.ndarray  # [2, H, W] assembly-averaged; scaled so mean power over fuel = 1
    power: np.ndarray  # [H, W] assembly-averaged fission rate, mean 1 over fuel assemblies


def _operator(D: np.ndarray, sr: np.ndarray, h_cm: float, bc: dict[str, str], gamma: float) -> sp.csc_matrix:
    """Loss operator -div(D grad) + Sr for one group on an [H, W] mesh (per unit area)."""
    H, W = D.shape
    idx = np.arange(H * W).reshape(H, W)
    diag = sr.astype(float).copy()
    rows, cols, vals = [], [], []
    for axis in (1, 0):  # x faces, then y faces
        a = np.s_[:, :-1] if axis == 1 else np.s_[:-1, :]
        b = np.s_[:, 1:] if axis == 1 else np.s_[1:, :]
        Da, Db = D[a], D[b]
        c = 2 * Da * Db / (Da + Db) / h_cm**2
        ia, ib = idx[a].ravel(), idx[b].ravel()
        c = c.ravel()
        np.add.at(diag, np.unravel_index(ia, (H, W)), c)
        np.add.at(diag, np.unravel_index(ib, (H, W)), c)
        rows += [ia, ib]
        cols += [ib, ia]
        vals += [-c, -c]
        lo, hi = ("west", "east") if axis == 1 else ("south", "north")
        for side, edge in ((lo, 0), (hi, -1)):
            if bc[side] == "reflective":
                continue
            if bc[side] != "vacuum":
                raise ValueError(f"unknown bc {bc[side]!r} on {side}")
            s = np.s_[:, edge] if axis == 1 else np.s_[edge, :]
            alpha = 2 * D[s] / h_cm
            diag[s] += gamma * alpha / (alpha + gamma) / h_cm
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(H * W, H * W))
    return (A + sp.diags(diag.ravel())).tocsc()


def solve(
    xs: np.ndarray,
    pitch_cm: float,
    fuel_mask: np.ndarray,
    n_sub: int = 4,
    bc: dict[str, str] = QUARTER_CORE_BC,
    gamma: float = MARSHAK,
) -> FVSolution:
    """Fundamental mode of the 2-group problem.

    xs [7, H, W] per-assembly XS in `model.physics.XS_FIELDS` order; fuel_mask [H, W] bool.
    """
    _, H, W = xs.shape
    fine = np.repeat(np.repeat(xs, n_sub, axis=1), n_sub, axis=2)
    D1, D2, sr1, sr2, nf1, nf2, s12 = fine
    h = pitch_cm / n_sub
    lu1 = splu(_operator(D1, sr1, h, bc, gamma))
    lu2 = splu(_operator(D2, sr2, h, bc, gamma))
    nf1, nf2, s12 = nf1.ravel(), nf2.ravel(), s12.ravel()

    def fluxes(src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p1 = lu1.solve(src)
        return p1, lu2.solve(s12 * p1)

    def fission_op(src: np.ndarray) -> np.ndarray:  # eigenvalue of this map is k_eff
        p1, p2 = fluxes(src)
        return nf1 * p1 + nf2 * p2

    n = fine[0].size
    op = LinearOperator((n, n), matvec=fission_op, dtype=float)
    vals, vecs = eigs(op, k=1, which="LM", v0=np.ones(n), tol=1e-10)
    k = float(vals[0].real)
    src = vecs[:, 0].real
    src *= np.sign(src.sum())
    p1, p2 = fluxes(src / k)

    def coarse(f: np.ndarray) -> np.ndarray:
        return f.reshape(H, n_sub, W, n_sub).mean(axis=(1, 3))

    flux = np.stack([coarse(p1.reshape(fine.shape[1:])), coarse(p2.reshape(fine.shape[1:]))])
    power = xs[4] * flux[0] + xs[5] * flux[1]  # fission-rate proxy, see agent.core docstring
    scale = power[fuel_mask].mean()
    return FVSolution(k_eff=k, flux=flux / scale, power=power / scale)
