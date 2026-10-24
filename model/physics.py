"""2-D, 2-group neutron diffusion physics shared by all models.

Steady-state k-eigenvalue problem (no upscatter, all fission neutrons born fast, chi = [1, 0]):

    -div(D1 grad phi1) + Sr1 phi1 = (nuSf1 phi1 + nuSf2 phi2) / k
    -div(D2 grad phi2) + Sr2 phi2 = S12 phi1

Sr_g is the removal XS (absorption + out-scatter, plus D_g B_z^2 axial buckling if the XS set
includes it). Units: cm, cm^-1.

Two residual implementations:
- `fv_residual`: batched cell-centered finite-volume residual on the assembly grid (FNO/CNN
  physics loss). Harmonic-mean interface D, same as a standard FV diffusion code.
- `TwoGroupDiffusion` / `DiffusionBC`: symbolic PhysicsNeMo PDEs for pointwise autodiff
  residuals (PINN).

Grid convention: tensors are [..., H, W] with index [i, j] = [y, x], row 0 at y = 0 ("south"),
column 0 at x = 0 ("west"). Quarter core: reflective on west/south, vacuum on east/north.
REVIEW (EECS): confirm this matches the datagen material_map orientation.
"""

from __future__ import annotations

import torch
from sympy import Function, Symbol

from physicsnemo.sym.eq.pde import PDE

# Per-material XS columns, in this order, everywhere in the model package.
XS_FIELDS = ("D1", "D2", "sig_r1", "sig_r2", "nu_sig_f1", "nu_sig_f2", "sig_s12")
N_XS = len(XS_FIELDS)

# Vacuum boundary coefficient gamma in D dphi/dn + gamma phi = 0.
# Marshak = 0.5; the IAEA 2-D benchmark specifies 0.4692.
MARSHAK = 0.5

QUARTER_CORE_BC = {"west": "reflective", "south": "reflective", "east": "vacuum", "north": "vacuum"}


def xs_maps(material_map: torch.Tensor, xs_table: torch.Tensor) -> torch.Tensor:
    """Per-cell XS fields. material_map [B, H, W] long, xs_table [n_mat, N_XS] -> [B, N_XS, H, W]."""
    return xs_table[material_map].permute(0, 3, 1, 2)


def _leakage(
    phi: torch.Tensor,
    D: torch.Tensor,
    h_cm: float,
    bc: dict[str, str],
    gamma: float,
    void: torch.Tensor | None,
) -> torch.Tensor:
    """FV net leakage per unit area, -div(D grad phi), for one group. phi, D: [B, H, W]."""
    out = torch.zeros_like(phi)
    for axis in (-1, -2):  # x (columns), y (rows)
        # Interior faces between cell c and its +1 neighbour along `axis`.
        a = phi.narrow(axis, 0, phi.shape[axis] - 1)
        b = phi.narrow(axis, 1, phi.shape[axis] - 1)
        Da = D.narrow(axis, 0, D.shape[axis] - 1)
        Db = D.narrow(axis, 1, D.shape[axis] - 1)
        D_face = 2 * Da * Db / (Da + Db).clamp_min(1e-12)
        J = D_face * (a - b) / h_cm  # current from a to b
        if void is not None:
            va = void.narrow(axis, 0, void.shape[axis] - 1)
            vb = void.narrow(axis, 1, void.shape[axis] - 1)
            # Face between fuel/reflector and void: vacuum condition on the non-void side.
            J = torch.where(va | vb, torch.zeros_like(J), J)
            out.narrow(axis, 0, J.shape[axis]).add_(
                torch.where(vb & ~va, _vacuum_current(a, Da, h_cm, gamma), torch.zeros_like(J))
            )
            out.narrow(axis, 1, J.shape[axis]).add_(
                torch.where(va & ~vb, _vacuum_current(b, Db, h_cm, gamma), torch.zeros_like(J))
            )
        out.narrow(axis, 0, J.shape[axis]).add_(J)
        out.narrow(axis, 1, J.shape[axis]).sub_(J)

        # Outer boundary faces.
        lo, hi = ("west", "east") if axis == -1 else ("south", "north")
        for side, idx in ((lo, 0), (hi, phi.shape[axis] - 1)):
            if bc[side] == "vacuum":
                c = phi.narrow(axis, idx, 1)
                out.narrow(axis, idx, 1).add_(_vacuum_current(c, D.narrow(axis, idx, 1), h_cm, gamma))
            elif bc[side] != "reflective":
                raise ValueError(f"unknown bc {bc[side]!r} on {side}")
    out = out / h_cm
    if void is not None:
        out = out.masked_fill(void, 0.0)
    return out


def _vacuum_current(phi_c: torch.Tensor, D: torch.Tensor, h_cm: float, gamma: float) -> torch.Tensor:
    """Outgoing current through a face with D dphi/dn + gamma phi = 0, half a cell from phi_c."""
    a = 2 * D / h_cm
    return gamma * a * phi_c / (a + gamma)


def fv_residual(
    flux: torch.Tensor,
    k_eff: torch.Tensor,
    xs: torch.Tensor,
    h_cm: float,
    bc: dict[str, str] = QUARTER_CORE_BC,
    gamma: float = MARSHAK,
    void: torch.Tensor | None = None,
) -> torch.Tensor:
    """Two-group FV residual [B, 2, H, W] (cm^-1 x flux units).

    flux [B, 2, H, W] in physical (not per-group normalized) units; k_eff [B] absolute k, not Δk.
    xs [B, N_XS, H, W] from `xs_maps`. void [B, H, W] bool marks cells outside the core.
    """
    D1, D2, sr1, sr2, nf1, nf2, s12 = xs.unbind(1)
    p1, p2 = flux.unbind(1)
    fission = nf1 * p1 + nf2 * p2
    r1 = _leakage(p1, D1, h_cm, bc, gamma, void) + sr1 * p1 - fission / k_eff.view(-1, 1, 1)
    r2 = _leakage(p2, D2, h_cm, bc, gamma, void) + sr2 * p2 - s12 * p1
    r = torch.stack([r1, r2], 1)
    if void is not None:
        r = r.masked_fill(void.unsqueeze(1), 0.0)
    return r


def physics_loss(flux, k_eff, xs, h_cm, bc=QUARTER_CORE_BC, gamma=MARSHAK, void=None) -> torch.Tensor:
    """Scale-free mean-square FV residual: ||r||^2 / ||fission source||^2, averaged over batch.

    Invariant to the overall flux amplitude (the eigenproblem fixes shape only), so it can be
    combined with a data loss regardless of how the flux is normalized.
    """
    r = fv_residual(flux, k_eff, xs, h_cm, bc, gamma, void)
    src = xs[:, 4] * flux[:, 0] + xs[:, 5] * flux[:, 1]
    num = r.pow(2).flatten(1).sum(1)
    den = src.pow(2).flatten(1).sum(1).clamp_min(1e-30)
    return (num / den).mean()


def rayleigh_k(flux, xs, h_cm, bc=QUARTER_CORE_BC, gamma=MARSHAK, void=None) -> torch.Tensor:
    """k implied by a flux shape via fast-group balance: k = sum(F phi) / sum(L1 phi1 + Sr1 phi1).

    Useful as a physics check on predicted flux independent of the model's k head.
    """
    D1, _, sr1, _, nf1, nf2, _ = xs.unbind(1)
    p1, p2 = flux.unbind(1)
    loss1 = _leakage(p1, D1, h_cm, bc, gamma, void) + sr1 * p1
    fission = nf1 * p1 + nf2 * p2
    if void is not None:
        fission = fission.masked_fill(void, 0.0)
        loss1 = loss1.masked_fill(void, 0.0)
    return fission.flatten(1).sum(1) / loss1.flatten(1).sum(1)


class TwoGroupDiffusion(PDE):
    """Symbolic 2-group diffusion PDE for PhysicsNeMo's `PhysicsInformer`.

    Inputs (all [N, 1] functions of (x, y)): phi1, phi2, and the XS fields in XS_FIELDS; `k` is a
    scalar input broadcast to [N, 1]. XS fields must be differentiable functions of the
    coordinates (D_g gradients enter via the conservative form).
    """

    name = "TwoGroupDiffusion"

    def __init__(self):
        self.dim = 2
        x, y = Symbol("x"), Symbol("y")
        phi1, phi2 = Function("phi1")(x, y), Function("phi2")(x, y)
        D1, D2, sr1, sr2, nf1, nf2, s12 = (Function(n)(x, y) for n in XS_FIELDS)
        k = Symbol("k")

        def div_D_grad(D, phi):
            return (D * phi.diff(x)).diff(x) + (D * phi.diff(y)).diff(y)

        self.equations = {
            "diffusion_fast": -div_D_grad(D1, phi1) + sr1 * phi1 - (nf1 * phi1 + nf2 * phi2) / k,
            "diffusion_thermal": -div_D_grad(D2, phi2) + sr2 * phi2 - s12 * phi1,
        }


class DiffusionBC(PDE):
    """Boundary residual D_g dphi_g/dn + gamma phi_g = 0 (gamma = 0 -> reflective).

    Inputs: phi1, phi2, D1, D2, normal_x, normal_y, gamma (outward normal; [N, 1] each).
    """

    name = "DiffusionBC"

    def __init__(self):
        self.dim = 2
        x, y = Symbol("x"), Symbol("y")
        nx, ny, gamma = Symbol("normal_x"), Symbol("normal_y"), Symbol("gamma")
        self.equations = {}
        for g in (1, 2):
            phi, D = Function(f"phi{g}")(x, y), Function(f"D{g}")(x, y)
            self.equations[f"bc_{g}"] = D * (nx * phi.diff(x) + ny * phi.diff(y)) + gamma * phi


def k_infinity(xs_row: torch.Tensor) -> torch.Tensor:
    """Infinite-medium k for one material row [N_XS]: (nuSf1 + nuSf2 S12/Sr2) / Sr1."""
    _, _, sr1, sr2, nf1, nf2, s12 = xs_row.unbind(-1)
    return (nf1 + nf2 * s12 / sr2) / sr1
