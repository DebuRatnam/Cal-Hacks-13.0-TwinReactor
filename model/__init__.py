"""Surrogate model: data loading, architectures, training, evaluation."""

from __future__ import annotations


def build_model(name: str, in_channels: int, n_groups: int = 2, **kw):
    """Grid surrogate by name: "fno" | "cnn" | "unet" (shape-only placeholder).

    The PINN is per-layout, not a grid surrogate; see `model.pinn.PINNProblem`.
    """
    if name == "fno":
        from model.grid_models import SurrogateFNO

        return SurrogateFNO(in_channels, n_groups, **kw)
    if name == "cnn":
        from model.grid_models import SurrogateCNN

        return SurrogateCNN(in_channels, n_groups, **kw)
    if name == "unet":
        from model.unet import SurrogateUNet

        return SurrogateUNet(in_channels, n_groups, **kw)
    raise ValueError(f"unknown model {name!r}")
