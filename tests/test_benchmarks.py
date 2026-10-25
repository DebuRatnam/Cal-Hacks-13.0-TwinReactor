"""Regression: reproduce published 2-group benchmark k_eff with ndiffusion before generating data."""

import json
from pathlib import Path

import numpy as np
import pytest

nd = pytest.importorskip("ndiffusion")


def test_iaea_2d_keff():
    from datagen.ndiffusion_gen import solve_case

    config = Path(__file__).parents[1] / "datagen/configs/iaea_explicit_examples.json"
    cfg = json.loads(config.read_text())
    case = cfg["cases"][0]
    params = {
        key: cfg[key] for key in ("xs_table", "rodded_xs_table", "pitch_cm", "bc", "boundary_gamma")
    }
    params["rod_state"] = case["rod_state"]
    result = solve_case(case["material_map"], params)
    # Mesh-dependent eigenvalue: allow 50 pcm against the published reference.
    assert abs(result.k_eff - nd.materials.IAEA.reference_keff) * 1e5 < 50
    assert result.solver_info["balance_residual"] < 1e-6
    assert np.isclose(result.flux.sum() * cfg["pitch_cm"] ** 2, 1)


@pytest.mark.skip(reason="TODO: wire up ndiffusion's built-in BIBLIS benchmark")
def test_biblis_keff(): ...
