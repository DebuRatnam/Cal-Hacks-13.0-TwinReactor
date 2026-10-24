"""Baseline U-Net: material map -> flux [G,H,W] + power [H,W], with a k_eff regression head."""

from __future__ import annotations

import torch
from torch import nn


class SurrogateUNet(nn.Module):
    def __init__(self, in_channels: int, n_groups: int = 2, width: int = 32):
        super().__init__()
        self.n_groups = n_groups
        # TODO: real encoder/decoder. Placeholder keeps shapes right for end-to-end testing.
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.field_head = nn.Conv2d(width, n_groups + 1, 1)
        self.k_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.body(x)
        fields = self.field_head(h)
        return {
            "flux": fields[:, : self.n_groups],
            "power": fields[:, self.n_groups],
            "k": self.k_head(h).squeeze(-1),
        }
