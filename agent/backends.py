"""Prediction backends the agent's tools call.

A backend maps per-assembly XS [7, H, W] to k_eff, flux, and power on the assembly grid. Today the
only one is the stand-in FV solver. When a trained checkpoint exists, add a `SurrogateBackend`
here (load model + normalization stats once, run `build_model(...)` on the XS channels,
de-normalize) and select it in `default_backend()`; the tools do not change.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from agent.fv_solver import FVSolution, solve


class Backend(Protocol):
    name: str  # shown to the scientist with every result

    def predict(self, xs: np.ndarray, pitch_cm: float, fuel_mask: np.ndarray) -> FVSolution: ...


class FVReferenceBackend:
    """Deterministic FV diffusion solve, n_sub x n_sub mesh cells per assembly."""

    # n_sub=16: ~7 pcm and ~0.6% peak power from the mesh-converged default core, ~0.2 s/solve.
    def __init__(self, n_sub: int = 16):
        self.n_sub = n_sub
        self.name = f"fv-diffusion-reference (stand-in, {n_sub}x{n_sub} mesh/assembly; ML surrogate not trained yet)"

    def predict(self, xs: np.ndarray, pitch_cm: float, fuel_mask: np.ndarray) -> FVSolution:
        return solve(xs, pitch_cm, fuel_mask, n_sub=self.n_sub)


def default_backend() -> Backend:
    return FVReferenceBackend()
