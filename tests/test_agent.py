"""Offline tests for the agent's tools on the trained surrogate (no API calls)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("anthropic")
pytest.importorskip("torch")
pytest.importorskip("physicsnemo")
if not (Path(__file__).resolve().parent.parent / "checkpoints" / "fno_softplus.pt").exists():
    pytest.skip("trained checkpoint not present", allow_module_level=True)

from anthropic.lib.tools import ToolError

from agent.tools import Session, make_tools


@pytest.fixture()
def session():
    return Session()


@pytest.fixture()
def tools(session):
    return {t.name: t for t in make_tools(session)}


def test_describe_core_reports_current_layout(tools, session):
    d = json.loads(tools["describe_core"].call({}))
    assert d["current_layout_material_ids"] == session.service.default_layout()
    assert set(d["rod_banks_row_col"]) == {str(b) for b in range(1, 9)}


def test_rod_insertion_monotone(tools, session):
    cfgs = [{"label": f"B3_{f}", "rods": {"3": f}} for f in (0.0, 1.0)]
    out = json.loads(tools["evaluate_configurations"].call({"configurations": cfgs}))["results"]
    assert out[0]["k_eff"] > out[1]["k_eff"]
    assert out[1]["delta_rho_vs_B3_0.0_pcm"] < 0
    assert session.new_labels == ["B3_0.0", "B3_1.0"]


def test_base_core_and_material_changes(tools, session):
    session.base_boron_ppm = 2000.0
    out = json.loads(
        tools["evaluate_configurations"].call(
            {
                "configurations": [
                    {"label": "base"},
                    {"label": "fresh centre", "material_changes": [{"row": 0, "col": 0, "material_id": 3}]},
                ]
            }
        )
    )["results"]
    assert out[0]["boron_ppm"] == 2000.0
    assert session.results["fresh centre"].layout[0][0] == 3


def test_get_map_relative(tools):
    tools["evaluate_configurations"].call(
        {"configurations": [{"label": "ARO"}, {"label": "B4", "rods": {"4": 1}}]}
    )
    m = json.loads(tools["get_map"].call({"label": "B4", "quantity": "thermal_flux", "relative_to": "ARO"}))["map"]
    assert m[0][0] < -20  # thermal flux depressed in the rodded centre assembly
    assert m[8][8] is None  # outside the core


def test_verify_with_reference(tools):
    tools["evaluate_configurations"].call({"configurations": [{"label": "ARO"}]})
    v = json.loads(tools["verify_with_reference"].call({"label": "ARO"}))
    assert abs(v["surrogate_k_error_pcm"]) < 1500
    assert v["reference_seconds"] > 0


@pytest.mark.parametrize(
    "cfg",
    [
        [{"label": "x", "rods": {"9": 1}}],
        [{"label": "x", "rods": {"1": 1.5}}],
        [{"label": "x", "boron_ppm": 3000}],
        [{"label": "x", "material_changes": [{"row": 8, "col": 0, "material_id": 0}]}],  # reflector
        [{"label": "x", "material_changes": [{"row": 0, "col": 0, "material_id": 9}]}],
        [{"label": "x"}, {"label": "x"}],
    ],
)
def test_bad_configurations_raise_tool_error(tools, cfg):
    with pytest.raises(ToolError):
        tools["evaluate_configurations"].call({"configurations": cfg})
