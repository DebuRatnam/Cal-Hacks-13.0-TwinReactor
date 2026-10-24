"""Offline tests for the agent's solver and tools (no API calls)."""

import json

import numpy as np
import pytest

pytest.importorskip("anthropic")

from agent.core import Core
from agent.fv_solver import solve
from agent.tools import Session, make_tools


@pytest.fixture(scope="module")
def tools():
    return {t.name: t for t in make_tools(Session())}


def test_solver_matches_model_fv_residual():
    """On the assembly mesh (n_sub=1) the solver satisfies model.physics.fv_residual."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("physicsnemo")
    from model.physics import fv_residual

    core = Core()
    xs, _ = core.build_xs({"C": 1.0})
    s = solve(xs, core.pitch_cm, core.fuel_mask(), n_sub=1)
    r = fv_residual(torch.tensor(s.flux)[None], torch.tensor([s.k_eff]), torch.tensor(xs)[None], core.pitch_cm)
    assert float(r.abs().max()) < 1e-6 * float(np.abs(xs[5] * s.flux[1]).max())


def test_default_core_symmetric_and_normalized():
    core = Core()
    xs, _ = core.build_xs({})
    m = core.fuel_mask()
    s = solve(xs, core.pitch_cm, m, n_sub=4)
    assert np.allclose(s.power, s.power.T, atol=1e-10)  # diagonal-symmetric input
    assert abs(s.power[m].mean() - 1) < 1e-12
    assert np.all(s.power[~m] == 0) and np.all(s.flux > 0)


def test_rod_insertion_monotone(tools):
    cfgs = [{"label": f"B{f}", "rods": {"B": f}} for f in (0.0, 0.5, 1.0)]
    out = json.loads(tools["evaluate_configurations"].call({"configurations": cfgs}))["results"]
    k = [r["k_eff"] for r in out]
    assert k[0] > k[1] > k[2]
    assert out[2]["delta_rho_vs_B0.0_pcm"] < out[1]["delta_rho_vs_B0.0_pcm"] < 0


def test_get_map_relative(tools):
    tools["evaluate_configurations"].call({"configurations": [{"label": "ARO"}, {"label": "A", "rods": {"A": 1}}]})
    m = json.loads(tools["get_map"].call({"label": "A", "quantity": "thermal_flux", "relative_to": "ARO"}))["map"]
    assert m[0][0] < -50  # thermal flux depressed in the rodded assembly
    assert m[0][-1] is None  # reflector excluded


@pytest.mark.parametrize(
    "cfg",
    [
        [{"label": "x", "rods": {"Z": 1}}],
        [{"label": "x", "rods": {"A": 1.5}}],
        [{"label": "x", "material_changes": [{"row": 0, "col": 0, "material": "unobtainium"}]}],
        [{"label": "x"}, {"label": "x"}],
    ],
)
def test_bad_configurations_raise_tool_error(tools, cfg):
    from anthropic.lib.tools import ToolError

    with pytest.raises(ToolError):
        tools["evaluate_configurations"].call({"configurations": cfg})
