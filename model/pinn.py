"""Physics-informed neural network for one 2-D, 2-group core layout.

Classic PINN: an MLP (physicsnemo.models.mlp.FullyConnected) maps (x, y) [cm] -> (phi1, phi2),
with k_eff as a trainable scalar. Trained on the PDE residual (PhysicsNeMo `PhysicsInformer`,
autodiff) + boundary residual + a normalization constraint (the eigenproblem only fixes the flux
shape). Optionally add flux data points from a reference solve.

Unlike FNO/CNN this is not a surrogate over layouts: it solves one layout per training run.
Use it as a mesh-free physics baseline / for inverse problems, not for the instant UI path.

REVIEW (EECS): material XS are piecewise constant per assembly; they are smoothed with sigmoids of
width `interface_width_cm` so D has a gradient for the conservative form. This approximates the
interface current-continuity condition; check k sensitivity to the width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from physicsnemo.models.mlp import FullyConnected
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

from model.physics import MARSHAK, QUARTER_CORE_BC, XS_FIELDS, DiffusionBC, TwoGroupDiffusion


class SmoothXSField(nn.Module):
    """Differentiable XS fields from an assembly map: coords [N, 2] cm -> {name: [N, 1]}."""

    def __init__(
        self,
        material_map: np.ndarray,
        xs_table: torch.Tensor,
        pitch_cm: float,
        interface_width_cm: float = 0.5,
        void_id: int | None = None,
    ):
        super().__init__()
        H, W = material_map.shape
        ii, jj = np.nonzero(material_map != void_id) if void_id is not None else np.indices((H, W)).reshape(2, -1)
        self.register_buffer("x0", torch.tensor(jj * pitch_cm, dtype=torch.float32))
        self.register_buffer("y0", torch.tensor(ii * pitch_cm, dtype=torch.float32))
        self.register_buffer("xs", xs_table[torch.as_tensor(material_map[ii, jj])].float())  # [n_cell, N_XS]
        self.pitch_cm = pitch_cm
        self.w = interface_width_cm

    def forward(self, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        x, y = coords[:, :1], coords[:, 1:2]
        s = torch.sigmoid
        p, w = self.pitch_cm, self.w
        wt = (
            s((x - self.x0) / w) * s((self.x0 + p - x) / w) * s((y - self.y0) / w) * s((self.y0 + p - y) / w)
        )  # [N, n_cell]
        vals = (wt @ self.xs) / wt.sum(1, keepdim=True).clamp_min(1e-12)
        return {name: vals[:, i : i + 1] for i, name in enumerate(XS_FIELDS)}


@dataclass
class _Segment:
    x0: float
    y0: float
    dx: float  # segment direction * length
    dy: float
    nx: float  # outward normal
    ny: float
    gamma: float


def boundary_segments(
    material_map: np.ndarray, pitch_cm: float, bc: dict[str, str], gamma: float, void_id: int | None
) -> list[_Segment]:
    """Outer faces of the non-void region with their BC coefficient (0 = reflective)."""
    H, W = material_map.shape
    p = pitch_cm
    is_void = (material_map == void_id) if void_id is not None else np.zeros((H, W), bool)
    segs = []
    # (di, dj, side, face start offset (x, y), direction, normal)
    faces = [
        (0, -1, "west", (0, 0), (0, p), (-1, 0)),
        (0, 1, "east", (p, 0), (0, p), (1, 0)),
        (-1, 0, "south", (0, 0), (p, 0), (0, -1)),
        (1, 0, "north", (0, p), (p, 0), (0, 1)),
    ]
    for i in range(H):
        for j in range(W):
            if is_void[i, j]:
                continue
            for di, dj, side, (ox, oy), (ddx, ddy), (nx, ny) in faces:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if not is_void[ni, nj]:
                        continue
                    g = gamma  # face onto void -> vacuum
                else:
                    g = 0.0 if bc[side] == "reflective" else gamma
                segs.append(_Segment(j * p + ox, i * p + oy, ddx, ddy, nx, ny, g))
    return segs


class PINN2G(nn.Module):
    """MLP (x, y) -> (phi1, phi2) >= 0, plus trainable k_eff."""

    def __init__(self, domain_cm: tuple[float, float], layer_size: int = 64, num_layers: int = 4, k_init: float = 1.0):
        super().__init__()
        self.register_buffer("scale", torch.tensor(domain_cm, dtype=torch.float32))
        self.net = FullyConnected(
            in_features=2, out_features=2, layer_size=layer_size, num_layers=num_layers, activation_fn="tanh"
        )
        self.log_k = nn.Parameter(torch.tensor(float(np.log(k_init))))

    @property
    def k_eff(self) -> torch.Tensor:
        return self.log_k.exp()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords [N, 2] in cm -> flux [N, 2]."""
        return F.softplus(self.net(2 * coords / self.scale - 1))


class PINNProblem:
    """Collocation sampling + losses for one layout."""

    def __init__(
        self,
        material_map: np.ndarray,
        xs_table: torch.Tensor,
        pitch_cm: float,
        bc: dict[str, str] = QUARTER_CORE_BC,
        gamma: float = MARSHAK,
        void_id: int | None = None,
        interface_width_cm: float = 0.5,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.material_map = material_map
        self.pitch_cm = pitch_cm
        H, W = material_map.shape
        self.domain_cm = (W * pitch_cm, H * pitch_cm)
        self.xs_field = SmoothXSField(material_map, xs_table, pitch_cm, interface_width_cm, void_id).to(self.device)
        self.cells = np.argwhere(material_map != void_id) if void_id is not None else np.argwhere(np.ones((H, W)))
        segs = boundary_segments(material_map, pitch_cm, bc, gamma, void_id)
        self.segs = torch.tensor(
            [[s.x0, s.y0, s.dx, s.dy, s.nx, s.ny, s.gamma] for s in segs], dtype=torch.float32, device=self.device
        )
        self.pde = PhysicsInformer(
            ["diffusion_fast", "diffusion_thermal"], TwoGroupDiffusion(), grad_method="autodiff", device=self.device
        )
        self.bc = PhysicsInformer(["bc_1", "bc_2"], DiffusionBC(), grad_method="autodiff", device=self.device)

    def make_model(self, **kw) -> PINN2G:
        return PINN2G(self.domain_cm, **kw).to(self.device)

    def sample_interior(self, n: int, gen: torch.Generator) -> torch.Tensor:
        idx = torch.randint(len(self.cells), (n,), generator=gen)
        ij = torch.as_tensor(self.cells, dtype=torch.float32)[idx]
        u = torch.rand(n, 2, generator=gen)
        xy = (ij[:, [1, 0]] + u) * self.pitch_cm
        return xy.to(self.device)

    def sample_boundary(self, n: int, gen: torch.Generator) -> torch.Tensor:
        idx = torch.randint(len(self.segs), (n,), generator=gen).to(self.device)
        s = self.segs[idx]
        t = torch.rand(n, 1, generator=gen).to(self.device)
        xy = s[:, :2] + t * s[:, 2:4]
        return torch.cat([xy, s[:, 4:7]], 1)  # [N, 5]: x, y, nx, ny, gamma

    def losses(self, model: PINN2G, n_interior: int = 2048, n_boundary: int = 512, gen=None) -> dict[str, torch.Tensor]:
        gen = gen if gen is not None else torch.Generator().manual_seed(0)

        xy = self.sample_interior(n_interior, gen).requires_grad_(True)
        phi = model(xy)
        xs = self.xs_field(xy)
        k = model.k_eff.expand(n_interior, 1)
        res = self.pde.forward({"coordinates": xy, "phi1": phi[:, :1], "phi2": phi[:, 1:], "k": k, **xs})
        src = xs["nu_sig_f1"] * phi[:, :1] + xs["nu_sig_f2"] * phi[:, 1:]
        src_ms = src.pow(2).mean().detach().clamp_min(1e-12)
        l_pde = (res["diffusion_fast"].pow(2).mean() + res["diffusion_thermal"].pow(2).mean()) / src_ms

        b = self.sample_boundary(n_boundary, gen)
        bxy = b[:, :2].clone().requires_grad_(True)
        bphi = model(bxy)
        bxs = self.xs_field(bxy)
        bres = self.bc.forward(
            {
                "coordinates": bxy,
                "phi1": bphi[:, :1],
                "phi2": bphi[:, 1:],
                "D1": bxs["D1"],
                "D2": bxs["D2"],
                "normal_x": b[:, 2:3],
                "normal_y": b[:, 3:4],
                "gamma": b[:, 4:5],
            }
        )
        phi_ms = phi.pow(2).mean().detach().clamp_min(1e-12)
        l_bc = (bres["bc_1"].pow(2).mean() + bres["bc_2"].pow(2).mean()) / phi_ms

        l_norm = (src.mean() - 1.0).pow(2)  # mean fission source = 1 fixes the amplitude
        return {"pde": l_pde, "bc": l_bc, "norm": l_norm, "total": l_pde + l_bc + l_norm}

    def evaluate_grid(self, model: PINN2G, n_per_cell: int = 4) -> np.ndarray:
        """Cell-averaged flux [2, H, W] (midpoint rule, n_per_cell^2 points per cell)."""
        H, W = self.material_map.shape
        n = n_per_cell
        u = (torch.arange(n) + 0.5) / n
        x = (torch.arange(W).repeat_interleave(n) + u.repeat(W)) * self.pitch_cm
        y = (torch.arange(H).repeat_interleave(n) + u.repeat(H)) * self.pitch_cm
        Y, X = torch.meshgrid(y, x, indexing="ij")
        with torch.no_grad():
            phi = model(torch.stack([X.flatten(), Y.flatten()], 1).to(self.device))
        return phi.T.reshape(2, H, n, W, n).mean(dim=(2, 4)).cpu().numpy()
