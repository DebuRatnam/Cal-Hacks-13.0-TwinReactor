"""Tools the agent calls. Each returns JSON text; numbers carry units in their key names.

All predictions come from the trained surrogate (`agent.backends.default_backend()`), on the same
core the web console edits. Configurations start from the session's base core (the layout and
boron the scientist has on screen) and apply rod positions and assembly swaps on top.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, NotRequired

import numpy as np
from anthropic import beta_tool
from anthropic.lib.tools import ToolError
from typing_extensions import TypedDict

from agent.backends import CoreService, LayoutError, default_backend

MAX_CONFIGS = 12


class MaterialChange(TypedDict):
    row: int
    col: int
    material_id: int


class Configuration(TypedDict):
    label: str
    rods: NotRequired[dict[str, float]]
    boron_ppm: NotRequired[float]
    material_changes: NotRequired[list[MaterialChange]]


@dataclass
class Result:
    label: str
    layout: list[list[int]]
    rods: dict[str, float]
    boron_ppm: float
    out: dict  # CoreService.predict output
    ref: dict | None = None  # CoreService.solve output, if verified


@dataclass
class Session:
    service: CoreService = field(default_factory=default_backend)
    base_layout: list[list[int]] | None = None  # None -> the service's default loading pattern
    base_rods: dict[str, float] = field(default_factory=dict)
    base_boron_ppm: float = 1000.0
    results: dict[str, Result] = field(default_factory=dict)
    new_labels: list[str] = field(default_factory=list)  # evaluated during the current turn

    def layout(self) -> list[list[int]]:
        return [list(r) for r in (self.base_layout or self.service.default_layout())]


def _rho_pcm(k: float) -> float:
    return (k - 1.0) / k * 1e5


def _arr(grid) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in grid], dtype=float)


def _loc(a: np.ndarray) -> list[int]:
    return [int(v) for v in np.unravel_index(np.nanargmax(a), a.shape)]


def _grid(a: np.ndarray, digits: int) -> list[list[float | None]]:
    return [[None if np.isnan(v) else round(float(v), digits) for v in row] for row in a]


def _summary(r: Result, first: Result | None) -> dict:
    o = r.out
    out = {
        "label": r.label,
        "rods_inserted_fraction": r.rods,
        "boron_ppm": r.boron_ppm,
        "k_eff": round(o["k_eff"], 5),
        "reactivity_pcm": round(_rho_pcm(o["k_eff"]), 1),
        "assembly_peaking": o["assembly_peaking"],
        "peak_assembly_row_col": o["assembly_peak_row_col"],
        "mesh_cell_peaking": o["node_peaking"],
        "power_map_assembly": _grid(_arr(o["power"]), 3),
    }
    if o["warnings"]:
        out["warnings"] = o["warnings"]
    if first is not None and first is not r:
        out[f"delta_rho_vs_{first.label}_pcm"] = round(
            _rho_pcm(o["k_eff"]) - _rho_pcm(first.out["k_eff"]), 1
        )
        p, p0 = _arr(o["power"]), _arr(first.out["power"])
        rel = (p - p0) / p0 * 100
        out[f"power_change_vs_{first.label}_pct"] = {
            "max_increase": round(float(np.nanmax(rel)), 2),
            "max_increase_row_col": _loc(rel),
            "max_decrease": round(float(np.nanmin(rel)), 2),
            "max_decrease_row_col": _loc(-rel),
        }
    return out


def make_tools(session: Session) -> list:
    svc = session.service

    @beta_tool
    def describe_core() -> str:
        """Describe the core the scientist is working on: grid and coordinate convention, the current layout (material ids), materials, rod banks, current boron, units, model accuracy, and limitations. Call this before the first evaluation in a conversation."""
        d = svc.describe()
        return json.dumps(
            {
                "grid": f"{d['grid'][0]}x{d['grid'][1]} quarter-core assemblies, pitch {d['assembly_pitch_cm']} cm",
                "coordinates": "layout[row][col]; row 0 / col 0 lie on the core-centre symmetry "
                "planes (reflective) and are half assemblies; the outer edge is vacuum. "
                "(0,0) is the core centre. A position stands for its mirror images in the full core.",
                "current_layout_material_ids": session.layout(),
                "layout_legend": "-1 = outside the core, "
                + ", ".join(f"{m['id']} = {m['name']}" for m in d["materials"]),
                "editable": "Only fuel positions can change, and only between fuel ids 0-5. "
                "Reflector and boundary positions are fixed (training geometry).",
                "rod_banks_row_col": d["banks"],
                "current_rods_inserted_fraction": session.base_rods,
                "current_boron_ppm": session.base_boron_ppm,
                "training_ranges": d["training_ranges"],
                "backend": d["model"],
                "units": d["units"],
                "limitations": [
                    (
                        f"Surrogate test error: k_eff MAE {d['model']['test_k_mae_pcm']:.0f} pcm "
                        f"(p95 {d['model']['test_k_p95_pcm']:.0f} pcm), assembly power ~"
                        f"{d['model']['test_power_rel_l2'] * 100:.1f}% relative L2. Differences "
                        "between similar configurations (rod worths) are more reliable than "
                        "absolute k_eff."
                    ),
                    "Rod banks were trained fully in or fully out; partial insertion extrapolates.",
                    (
                        "Fixed fuel 900 K and coolant 580 K: no temperature feedback. No axial "
                        "model, so no Fq; mesh-cell peaking is the finest radial measure."
                    ),
                    d["scope"],
                ],
            }
        )

    @beta_tool
    def evaluate_configurations(configurations: list[Configuration]) -> str:
        """Run the trained surrogate on one or more configurations and return k_eff, reactivity, peaking and assembly power maps for each. Put every configuration to be compared in a single call. The first configuration is the reference for delta-reactivity and power-change fields. Each configuration starts from the scientist's current core (layout and boron). Results are stored by label for get_map and verify_with_reference, and appear in the scientist's console.

        Args:
            configurations: Up to 12 configurations. Each has a unique `label`; optional `rods` mapping bank id ("1".."8") to inserted fraction 0..1 (banks not listed are withdrawn); optional `boron_ppm` (0-2000, default: the current core's boron); optional `material_changes` [{row, col, material_id}] swapping fuel assemblies.
        """
        if not configurations:
            raise ToolError("give at least one configuration")
        if len(configurations) > MAX_CONFIGS:
            raise ToolError(f"at most {MAX_CONFIGS} configurations per call")
        labels = [c["label"] for c in configurations]
        if len(set(labels)) != len(labels):
            raise ToolError(f"labels must be unique, got {labels}")
        out, first = [], None
        for cfg in configurations:
            layout = session.layout()
            for ch in cfg.get("material_changes", []):
                i, j = ch["row"], ch["col"]
                if not (0 <= i < len(layout) and 0 <= j < len(layout[0])):
                    raise ToolError(f"configuration {cfg['label']!r}: ({i}, {j}) outside the grid")
                layout[i][j] = ch["material_id"]
            rods = {str(b): float(f) for b, f in cfg.get("rods", {}).items() if f > 0}
            boron = float(cfg.get("boron_ppm", session.base_boron_ppm))
            try:
                pred = svc.predict(layout, rods, boron)
            except LayoutError as e:
                raise ToolError(f"configuration {cfg['label']!r}: {e}") from e
            r = Result(cfg["label"], layout, rods, boron, pred)
            session.results[r.label] = r
            if r.label not in session.new_labels:
                session.new_labels.append(r.label)
            first = first or r
            out.append({**_summary(r, first), "compute_ms": pred["compute_ms"]})
        return json.dumps({"backend": svc.describe_model_short(), "results": out})

    @beta_tool
    def get_map(
        label: str,
        quantity: Literal["power", "fast_flux", "thermal_flux", "material"],
        relative_to: str | None = None,
    ) -> str:
        """Return a full assembly map for an evaluated configuration, optionally as percent change against another evaluated configuration. Use for spatial questions (tilts, flux depressions, shadowing) that the summaries do not answer.

        Args:
            label: Label of a configuration already run with evaluate_configurations.
            quantity: Which map. Flux is in 1e-5 cm^-2 with the integral of (phi1 + phi2) over the quarter core = 1.
            relative_to: Optional second label; returns 100 * (label - relative_to) / relative_to.
        """
        for lab in (label, relative_to):
            if lab is not None and lab not in session.results:
                raise ToolError(f"no results for {lab!r}; evaluated: {sorted(session.results)}")
        r = session.results[label]
        if quantity == "material":
            return json.dumps({"label": label, "map": r.layout})
        key = {"power": "power", "fast_flux": "fast", "thermal_flux": "thermal"}[quantity]
        a = _arr(r.out[key])
        if relative_to is None:
            units = (
                "relative, mean 1 over fuel cells (assembly mean)"
                if key == "power"
                else r.out["flux_unit"]
            )
            return json.dumps({"label": label, "quantity": quantity, "units": units, "map": _grid(a, 4)})
        b = _arr(session.results[relative_to].out[key])
        rel = (a - b) / b * 100
        return json.dumps(
            {
                "label": label,
                "relative_to": relative_to,
                "quantity": quantity,
                "units": "percent change",
                "map": _grid(rel, 2),
            }
        )

    @beta_tool
    def verify_with_reference(label: str) -> str:
        """Re-run an evaluated configuration with the reference 2-group diffusion solver (ndiffusion, ~1 s) and report the surrogate's error. Use sparingly: for the configuration a conclusion rests on, when the scientist asks for confirmation, or when the surrogate result carries warnings.

        Args:
            label: Label of a configuration already run with evaluate_configurations.
        """
        if label not in session.results:
            raise ToolError(f"no results for {label!r}; evaluated: {sorted(session.results)}")
        r = session.results[label]
        try:
            ref = svc.solve(r.layout, r.rods, r.boron_ppm)
        except (LayoutError, RuntimeError) as e:
            raise ToolError(f"reference solve failed: {e}") from e
        r.ref = ref
        p, pr = _arr(r.out["power"]), _arr(ref["power"])
        err = (p - pr) / pr * 100
        return json.dumps(
            {
                "label": label,
                "reference": ref["source"],
                "k_eff_reference": round(ref["k_eff"], 5),
                "k_eff_surrogate": round(r.out["k_eff"], 5),
                "surrogate_k_error_pcm": round((r.out["k_eff"] - ref["k_eff"]) * 1e5, 1),
                "assembly_peaking_reference": ref["assembly_peaking"],
                "assembly_peaking_surrogate": r.out["assembly_peaking"],
                "assembly_power_error_pct": {
                    "max_abs": round(float(np.nanmax(np.abs(err))), 2),
                    "at_row_col": _loc(np.abs(err)),
                    "rms": round(float(np.sqrt(np.nanmean(err**2))), 2),
                },
                "reference_seconds": round(ref["compute_ms"] / 1000, 2),
            }
        )

    return [describe_core, evaluate_configurations, get_map, verify_with_reference]
