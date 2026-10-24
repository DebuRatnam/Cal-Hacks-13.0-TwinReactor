"""Grid surrogates (PhysicsNeMo backbones): core map [B, C, H, W] -> flux, power, k.

Both models share the output contract of `SurrogateUNet`:
    {"flux": [B, G, H, W], "power": [B, H, W], "k": [B]}
with "k" = k_eff - 1 (the dataset target), and flux/power in whatever normalized units the
training loop uses. De-normalize flux before calling `model.physics.physics_loss`.

Input channels are up to the training loop, typically one-hot material ids and/or the per-cell
XS fields from `model.physics.xs_maps` (XS channels generalize across fuel types).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from physicsnemo.models.fno import FNO
from physicsnemo.models.pix2pix import Pix2Pix


def _split(out: torch.Tensor, n_groups: int) -> dict[str, torch.Tensor]:
    """Channels: [flux_1..flux_G, power, k-field]; k = spatial mean of the k-field."""
    return {
        "flux": out[:, :n_groups],
        "power": out[:, n_groups],
        "k": out[:, n_groups + 1].mean(dim=(-2, -1)),
    }


class SurrogateFNO(nn.Module):
    """Fourier Neural Operator (physicsnemo.models.fno.FNO), 2-D.

    num_fno_modes must be <= (H + padding) // 2 + 1; the default suits a 9x9 assembly grid.
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


class SurrogateCNN(nn.Module):
    """Conv encoder / ResNet blocks / decoder (physicsnemo.models.pix2pix.Pix2Pix), 2-D.

    Input is replicate-padded on the east/north sides up to a multiple of 2**n_downsampling,
    then cropped back, so any H x W works. West/south (symmetry lines) are left untouched.
    """

    def __init__(
        self,
        in_channels: int,
        n_groups: int = 2,
        conv_layer_size: int = 32,
        n_downsampling: int = 2,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.multiple = 2**n_downsampling
        self.net = Pix2Pix(
            in_channels=in_channels,
            out_channels=n_groups + 2,
            dimension=2,
            conv_layer_size=conv_layer_size,
            n_downsampling=n_downsampling,
            n_upsampling=n_downsampling,
            n_blocks=n_blocks,
            activation_fn="relu",
            batch_norm=False,
            padding_type="reflect",
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        H, W = x.shape[-2:]
        ph, pw = -H % self.multiple, -W % self.multiple
        out = self.net(F.pad(x, (0, pw, 0, ph), mode="replicate"))[..., :H, :W]
        return _split(out, self.n_groups)
