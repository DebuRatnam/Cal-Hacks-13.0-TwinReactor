"""2-D, 2-group neutron diffusion physics used by the FNO's physics loss and checks.

Steady-state k-eigenvalue problem (no upscatter, all fission neutrons born fast, chi = [1, 0]):

    -div(D1 grad phi1) + Sr1 phi1 = (nuSf1 phi1 + nuSf2 phi2) / k
    -div(D2 grad phi2) + Sr2 phi2 = S12 phi1

Sr_g is the removal XS (absorption + out-scatter, plus D_g B_z^2 axial buckling if the XS set
includes it). Units: cm, cm^-1.

`fv_residual`: batched cell-centered finite-volume residual on the numerical grid (FNO
physics loss). Harmonic-mean interface D, same as a standard FV diffusion code.

Confirmed datagen convention: tensors are [..., H, W], index [i, j] = [y, x].
Row 0 is the southernmost cell row; column 0 is west. Cell centers are
x=(j+0.5)*h, y=(i+0.5)*h; south/west faces are at zero. Quarter core: reflective
on west/south, vacuum on east/north and exposed stepped boundaries. See
tests/test_data_compatibility.py and docs/DATA_HANDOFF.md for verification.
"""

from __future__ import annotations

import torch

from datagen.physics_contract import MARSHAK, QUARTER_CORE_BC, XS_FIELDS

# Per-material XS columns, in this order, everywhere in the model package.
N_XS = len(XS_FIELDS)

# Vacuum boundary coefficient gamma in D dphi/dn + gamma phi = 0.
# Marshak = 0.5; the IAEA 2-D benchmark specifies 0.4692.


def xs_maps(material_map: torch.Tensor, xs_table: torch.Tensor) -> torch.Tensor:
    """Physics IDs [B,H,W] and matching table [n_mat,7] -> coefficient maps.

    For v3 data use batch['xs'] directly: semantic material IDs do not encode
    the case's boron and local rod states. compact_xs supplies matching physics
    IDs/table when that representation is needed.
    """
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
                out.narrow(axis, idx, 1).add_(
                    _vacuum_current(c, D.narrow(axis, idx, 1), h_cm, gamma)
                )
            elif bc[side] != "reflective":
                raise ValueError(f"unknown bc {bc[side]!r} on {side}")
    out = out / h_cm
    if void is not None:
        out = out.masked_fill(void, 0.0)
    return out


def _vacuum_current(
    phi_c: torch.Tensor, D: torch.Tensor, h_cm: float, gamma: float
) -> torch.Tensor:
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


def physics_loss(
    flux, k_eff, xs, h_cm, bc=QUARTER_CORE_BC, gamma=MARSHAK, void=None
) -> torch.Tensor:
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


def k_infinity(xs_row: torch.Tensor) -> torch.Tensor:
    """Infinite-medium k for one material row [N_XS]: (nuSf1 + nuSf2 S12/Sr2) / Sr1."""
    _, _, sr1, sr2, nf1, nf2, s12 = xs_row.unbind(-1)
    return (nf1 + nf2 * s12 / sr2) / sr1
