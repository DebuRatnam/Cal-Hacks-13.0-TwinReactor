"""Shape tests for the PhysicsNeMo FNO surrogate and analytic checks of the 2-group physics residuals."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("physicsnemo")

from model import build_model  # noqa: E402
from model.physics import (  # noqa: E402
    N_XS,
    fv_residual,
    k_infinity,
    physics_loss,
    rayleigh_k,
    xs_maps,
)

# IAEA-2D fuel 1 (with axial buckling folded into removal): D1, D2, Sr1, Sr2, nuSf1, nuSf2, S12
FUEL = torch.tensor([1.5, 0.4, 0.03 + 0.02 + 1.5 * 0.8e-4, 0.08 + 0.4 * 0.8e-4, 0.0, 0.135, 0.02])
ALL_REFLECTIVE = dict(west="reflective", east="reflective", south="reflective", north="reflective")


@pytest.mark.parametrize("hw", [(9, 9), (8, 11)])
def test_fno_shapes(hw):
    model = build_model("fno", in_channels=N_XS)
    out = model(torch.randn(3, N_XS, *hw))
    assert out["flux"].shape == (3, 2, *hw)
    assert out["power"].shape == (3, *hw)
    assert out["k"].shape == (3,)
    sum(v.sum() for v in out.values()).backward()


def test_fv_infinite_medium():
    """Flat flux in an all-reflective homogeneous core is exact with k = k_inf."""
    xs = xs_maps(torch.zeros(1, 9, 9, dtype=torch.long), FUEL[None])
    k_inf = k_infinity(FUEL)
    _, _, _, sr2, _, _, s12 = FUEL
    flux = torch.stack([torch.ones(9, 9), torch.full((9, 9), float(s12 / sr2))])[None]
    r = fv_residual(flux, k_inf[None], xs, h_cm=20.0, bc=ALL_REFLECTIVE)
    assert r.abs().max() < 1e-6
    assert torch.allclose(rayleigh_k(flux, xs, 20.0, bc=ALL_REFLECTIVE), k_inf, rtol=1e-6)


def test_fv_cosine_mode_converges():
    """Homogeneous quarter core, zero flux at x=y=L: phi ~ cos(Bx x) cos(By y). FV k -> analytic k."""
    L, n = 100.0, 50
    h = L / n
    D1, D2, sr1, sr2, nf1, nf2, s12 = FUEL.double()
    B2 = 2 * (math.pi / (2 * L)) ** 2
    ratio = s12 / (D2 * B2 + sr2)
    k_exact = (nf1 + nf2 * ratio) / (D1 * B2 + sr1)
    c = (torch.arange(n, dtype=torch.float64) + 0.5) * h
    mode = torch.outer(torch.cos(math.pi * c / (2 * L)), torch.cos(math.pi * c / (2 * L)))
    flux = torch.stack([mode, ratio * mode])[None]
    xs = xs_maps(torch.zeros(1, n, n, dtype=torch.long), FUEL.double()[None])
    # gamma huge -> phi = 0 on the outer face (Dirichlet).
    k_fv = rayleigh_k(flux, xs, h, gamma=1e9)
    assert abs(float(k_fv - k_exact)) * 1e5 < 50  # pcm
    assert physics_loss(flux, k_exact.view(1), xs, h, gamma=1e9) < 1e-4
