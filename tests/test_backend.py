"""Backend contract: UI layout -> model input, validation, and surrogate vs reference sanity."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("physicsnemo")
if not (Path(__file__).resolve().parent.parent / "checkpoints" / "fno_softplus.pt").exists():
    pytest.skip("trained checkpoint not present", allow_module_level=True)

from backend.surrogate import CoreService, LayoutError


@pytest.fixture(scope="module")
def svc():
    return CoreService()


def test_describe_matches_geometry(svc):
    d = svc.describe()
    H, W = d["grid"]
    assert (H, W) == (9, 9)
    assert len(d["banks"]) == 8
    layout = d["default_layout"]
    for i in range(H):
        for j in range(W):
            kind = d["site"][i][j]
            v = layout[i][j]
            assert (kind == "void") == (v == -1)
            assert (kind == "reflector") == (v == svc.reflector_id)


def test_predict_outputs_and_physics_checks(svc):
    layout = svc.default_layout()
    out = svc.predict(layout, {}, 1000)
    assert 0.8 < out["k_eff"] < 1.4
    assert out["warnings"] == []  # in-distribution input; negative power is clipped, not warned
    for i, row in enumerate(out["power"]):
        for j, v in enumerate(row):
            kind = svc.site[i, j]
            assert (v is None) == (kind != "fuel")  # power only in fuel
    assert all(v is None or v >= 0 for row in out["thermal"] for v in row)
    assert all(v is None or v >= 0 for row in out["mesh"]["power"] for v in row)
    assert out["raw_normalization"]["power_min_before_clip"] <= out["mesh"]["power"][0][0]
    mesh = out["mesh"]
    assert all(len(mesh[q]) == 68 and all(len(r) == 68 for r in mesh[q]) for q in mesh)
    fuel_cells = [v for row in mesh["power"] for v in row if v is not None]
    assert abs(sum(fuel_cells) / len(fuel_cells) - 1) < 1e-3  # mean 1 over fuel cells
    assert max(fuel_cells) == pytest.approx(out["node_peaking"], abs=1e-4)


def test_rod_insertion_lowers_k(svc):
    layout = svc.default_layout()
    out = svc.predict(layout, {}, 1000)["k_eff"]
    rodded = svc.predict(layout, {"4": 1.0, "8": 1.0}, 1000)["k_eff"]
    assert rodded < out


def test_ood_warnings(svc):
    out = svc.predict(svc.default_layout(), {"1": 0.5}, 750)
    assert len(out["warnings"]) == 2


@pytest.mark.parametrize(
    "change, rods, boron",
    [
        ((8, 0, 0), {}, 1000),  # reflector -> fuel
        ((8, 8, 0), {}, 1000),  # void -> fuel
        (None, {"9": 1.0}, 1000),  # unknown bank
        (None, {}, 2500),  # boron outside the tables
    ],
)
def test_rejects_inputs_outside_model(svc, change, rods, boron):
    layout = svc.default_layout()
    if change:
        i, j, v = change
        layout[i][j] = v
    with pytest.raises(LayoutError):
        svc.predict(layout, rods, boron)


def test_chat_endpoint_returns_agent_configs(monkeypatch):
    """/chat plumbing with Claude replaced by a scripted turn that calls the real tools."""
    pytest.importorskip("anthropic")
    from fastapi.testclient import TestClient

    import agent.agent
    from agent.tools import Session, make_tools
    from backend import app as app_module

    class ScriptedAgent:
        def __init__(self):
            self.session = Session()
            self.tools = {t.name: t for t in make_tools(self.session)}

        def set_base(self, layout, rods, boron_ppm):
            self.session.base_layout, self.session.base_boron_ppm = layout, boron_ppm

        def ask(self, text, on_tool_call):
            self.session.new_labels.clear()
            args = {"configurations": [{"label": "ARO"}, {"label": "Bank 3 in", "rods": {"3": 1}}]}
            on_tool_call("evaluate_configurations", args)
            self.tools["evaluate_configurations"].call(args)
            return "| Config | k_eff |\n|---|---|\n| ARO | 1.1 |"

    monkeypatch.setattr(agent.agent, "CoreAgent", ScriptedAgent)
    app_module._agents.clear()
    client = TestClient(app_module.app)
    layout = CoreService().default_layout()
    r = client.post("/chat", json={"session_id": "s", "message": "bank 3 worth?", "core": {"layout": layout, "rods": {}, "boron_ppm": 1500}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [c["label"] for c in body["configs"]] == ["ARO", "Bank 3 in"]
    assert body["configs"][0]["boron_ppm"] == 1500
    assert body["configs"][1]["result"]["k_eff"] < body["configs"][0]["result"]["k_eff"]
    assert body["tool_calls"][0]["name"] == "evaluate_configurations"


def test_reference_agrees_roughly(svc):
    layout = svc.default_layout()
    pred = svc.predict(layout, {}, 1000)
    ref = svc.solve(layout, {}, 1000)
    assert abs(pred["k_eff"] - ref["k_eff"]) * 1e5 < 1500  # pcm; test-set p95 is ~540
