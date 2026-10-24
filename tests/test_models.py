"""Shape tests for the PhysicsNeMo surrogates and analytic checks of the 2-group physics residuals."""

import math

import numpy as np
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


@pytest.mark.parametrize("name", ["fno", "cnn"])
@pytest.mark.parametrize("hw", [(9, 9), (8, 11)])
def test_grid_model_shapes(name, hw):
    model = build_model(name, in_channels=N_XS)
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


def test_pinn_residual_on_analytic_mode():
    """PhysicsNeMo autodiff residual of the analytic cosine mode is ~0 away from cell edges."""
    from physicsnemo.sym.eq.phy_informer import PhysicsInformer

    from model.physics import TwoGroupDiffusion

    L = 100.0
    D1, D2, sr1, sr2, nf1, nf2, s12 = FUEL.tolist()
    B = math.pi / (2 * L)
    ratio = s12 / (D2 * 2 * B**2 + sr2)
    k_exact = (nf1 + nf2 * ratio) / (D1 * 2 * B**2 + sr1)

    xy = (torch.rand(256, 2, dtype=torch.float64) * L).requires_grad_(True)
    mode = torch.cos(B * xy[:, :1]) * torch.cos(B * xy[:, 1:])
    ones = torch.ones_like(mode)
    xs = {n: v * ones for n, v in zip(("D1", "D2", "sig_r1", "sig_r2", "nu_sig_f1", "nu_sig_f2", "sig_s12"), FUEL.tolist())}
    # D must depend on coordinates for the autodiff graph; add an exact-zero dependence.
    xs["D1"] = xs["D1"] + 0 * xy[:, :1]
    xs["D2"] = xs["D2"] + 0 * xy[:, :1]
    pi = PhysicsInformer(["diffusion_fast", "diffusion_thermal"], TwoGroupDiffusion(), grad_method="autodiff")
    res = pi.forward({"coordinates": xy, "phi1": mode, "phi2": ratio * mode, "k": k_exact * ones, **xs})
    assert res["diffusion_fast"].abs().max() < 1e-10
    assert res["diffusion_thermal"].abs().max() < 1e-10


def test_pinn_problem_losses_backprop():
    from model.pinn import PINNProblem

    mat = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 2]])  # 2 = void corner
    table = torch.stack([FUEL, FUEL * torch.tensor([1, 1, 1, 1, 0, 0, 1.0])])
    prob = PINNProblem(mat, table, pitch_cm=20.0, void_id=2)
    model = prob.make_model(layer_size=16, num_layers=2)
    losses = prob.losses(model, n_interior=128, n_boundary=64)
    losses["total"].backward()
    assert model.log_k.grad is not None
    assert all(torch.isfinite(v) for v in losses.values())
    assert prob.evaluate_grid(model, n_per_cell=2).shape == (2, 3, 3)
