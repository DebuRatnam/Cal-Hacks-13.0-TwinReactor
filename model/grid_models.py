"""Grid surrogate (PhysicsNeMo FNO backbone): core map [B, C, H, W] -> flux, power, k.

Output contract:
    {"flux": [B, G, H, W], "power": [B, H, W], "k": [B]}
with "k" = k_eff - 1 (the dataset target), and flux/power in whatever normalized units the
training loop uses. De-normalize flux before calling `model.physics.physics_loss`.

Input channels are the per-cell XS fields (`batch["xs"]`, 7 channels), which make boron and
rod state visible to the model.
"""

from __future__ import annotations

import torch
from physicsnemo.models.fno import FNO
from torch import nn


def _split(out: torch.Tensor, n_groups: int) -> dict[str, torch.Tensor]:
    """Channels: [flux_1..flux_G, power, k-field]; k = spatial mean of the k-field."""
    return {
        "flux": torch.nn.functional.softplus(out[:, :n_groups]),
        "power": out[:, n_groups],
        "k": out[:, n_groups + 1].mean(dim=(-2, -1)),
    }


class SurrogateFNO(nn.Module):
    """Fourier Neural Operator (physicsnemo.models.fno.FNO), 2-D.

    num_fno_modes must be <= (H + padding) // 2 + 1.
    """

    def __init__(
        self,
        in_channels: int,
        n_groups: int = 2,
        latent_channels: int = 32,
        num_fno_layers: int = 4,
        num_fno_modes: int = 6,
        padding: int = 4,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.net = FNO(
            in_channels=in_channels,
            out_channels=n_groups + 2,
            dimension=2,
            latent_channels=latent_channels,
            num_fno_layers=num_fno_layers,
            num_fno_modes=num_fno_modes,
            padding=padding,
            coord_features=True,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return _split(self.net(x), self.n_groups)
