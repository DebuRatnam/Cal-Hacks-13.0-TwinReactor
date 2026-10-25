"""Surrogate model: data loading, architecture, training, evaluation."""

from __future__ import annotations


def build_model(name: str, in_channels: int, n_groups: int = 2, **kw):
    """Grid surrogate by name. Only "fno" (PhysicsNeMo Fourier Neural Operator) is kept.

    The name argument stays so saved checkpoints (config["model"] == "fno") load unchanged.
    """
    if name == "fno":
        from model.grid_models import SurrogateFNO

        return SurrogateFNO(in_channels, n_groups, **kw)
    raise ValueError(f"unknown model {name!r}; only 'fno' is available")
