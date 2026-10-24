"""Tools the agent calls. Each returns JSON text; numbers carry units in their key names."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal, NotRequired

import numpy as np
from anthropic import beta_tool
from anthropic.lib.tools import ToolError
from typing_extensions import TypedDict

from agent.backends import Backend, default_backend
from agent.core import MATERIALS, Core
from agent.fv_solver import FVSolution

MAX_CONFIGS = 12


class MaterialChange(TypedDict):
    row: int
    col: int
    material: str


class Configuration(TypedDict):
    label: str
    rods: NotRequired[dict[str, float]]
    material_changes: NotRequired[list[MaterialChange]]


@dataclass
class Result:
    config: Configuration
    layout: list[list[str]]
    fuel_mask: np.ndarray
    sol: FVSolution


@dataclass
class Session:
    core: Core = field(default_factory=Core)
    backend: Backend = field(default_factory=default_backend)
    results: dict[str, Result] = field(default_factory=dict)


def _rho_pcm(k: float) -> float:
    return (k - 1.0) / k * 1e5


def _loc(a: np.ndarray, mask: np.ndarray) -> list[int]:
    return [int(v) for v in np.unravel_index(np.where(mask, a, -np.inf).argmax(), a.shape)]


def _grid(a: np.ndarray, digits: int) -> list[list[float | None]]:
    return [[None if np.isnan(v) else round(float(v), digits) for v in row] for row in a]


def _summary(r: Result, ref: Result | None) -> dict:
    s, m = r.sol, r.fuel_mask
    out = {
        "label": r.config["label"],
        "rods_inserted_fraction": r.config.get("rods", {}),
        "material_changes": r.config.get("material_changes", []),
        "k_eff": round(s.k_eff, 5),
        "reactivity_pcm": round(_rho_pcm(s.k_eff), 1),
        "assembly_radial_peaking": round(float(s.power[m].max()), 3),
        "peak_power_assembly_row_col": _loc(s.power, m),
        # Mean fuel thermal flux is not reported: with power normalized to 1 and the same nuSf2 in
        # every fuel type it is a constant. The spectral ratio carries the information.
        "fast_flux_fuel_mean": round(float(s.flux[0][m].mean()), 4),
        "thermal_to_fast_flux_ratio_fuel_mean": round(float((s.flux[1][m] / s.flux[0][m]).mean()), 4),
        "thermal_flux_max_row_col": _loc(s.flux[1], m),
        "thermal_flux_min_fuel_row_col": _loc(-s.flux[1], m),
        "power_map": _grid(np.where(m, s.power, np.nan), 3),
    }
    if ref is not None and ref is not r:
        out["delta_rho_vs_" + ref.config["label"] + "_pcm"] = round(_rho_pcm(s.k_eff) - _rho_pcm(ref.sol.k_eff), 1)
        both = m & ref.fuel_mask
        rel = np.zeros_like(s.power)
        rel[both] = (s.power[both] - ref.sol.power[both]) / ref.sol.power[both] * 100
        out["power_change_vs_" + ref.config["label"] + "_pct"] = {
            "max_increase": round(float(rel[both].max()), 2),
            "max_increase_row_col": _loc(rel, both),
            "max_decrease": round(float(rel[both].min()), 2),
            "max_decrease_row_col": _loc(-rel, both),
        }
    return out


def make_tools(session: Session) -> list:
    core = session.core

    @beta_tool
    def describe_core() -> str:
        """Describe the current core model: grid and coordinate convention, materials, rod banks, the prediction backend, units, and known limitations. Call this before the first evaluation in a conversation."""
        H, W = core.shape
        return json.dumps(
            {
                "grid": f"{H}x{W} assemblies, quarter core, pitch {core.pitch_cm} cm",
                "coordinates": "map[row][col]; row 0 / col 0 lie on the core-centre symmetry lines "
                "(reflective boundary along assembly faces); the last row/col face the vacuum "
                "boundary. Assembly (0,0) touches the core centre; by symmetry it stands for the "
                "central 2x2 assemblies of the full core, and a bank position stands for its mirror images.",
                "layout": core.layout,
                "materials": {
                    k: {"fuel": v["fuel"], "source": v["note"], "xs_cm-1": dict(zip(
                        ("D1_cm", "D2_cm", "sig_r1", "sig_r2", "nu_sig_f1", "nu_sig_f2", "sig_s12"),
                        np.round(v["xs"], 5).tolist()))}
                    for k, v in MATERIALS.items()
                },
                "rod_banks_row_col": core.banks,
                "rod_insertion": "fraction 0 (withdrawn, default) to 1 (fully inserted) per bank",
                "backend": session.backend.name,
                "outputs": {
                    "k_eff": "absolute multiplication factor",
                    "reactivity_pcm": "(k-1)/k * 1e5",
                    "power": "assembly-average fission rate, normalized to mean 1 over fuel assemblies",
                    "flux": "assembly-average 2-group scalar flux (group 1 fast, group 2 thermal), "
                    "same normalization as power (relative units)",
                    "assembly_radial_peaking": "max assembly power / mean (pin-level Fq, F_dH not available)",
                },
                "limitations": [
                    "Research surrogate output, not safety-grade.",
                    "Placeholder XS (IAEA 2-D benchmark), not the project's OpenMC-derived set.",
                    "Partial rod insertion is a volume-weighted 2-D XS blend; overstates partial-bank worth.",
                    "No boron, temperature, or burnup feedback in this model.",
                ],
            }
        )

    @beta_tool
    def evaluate_configurations(configurations: list[Configuration]) -> str:
        """Run the core model on one or more configurations and return k_eff, reactivity, power and flux summaries for each. Put every configuration the scientist wants compared in a single call. The first configuration is the reference for delta-reactivity and power-change fields. Results are stored by label for get_map.

        Args:
            configurations: Up to 12 configurations. Each has a unique `label`, optional `rods` mapping bank name to inserted fraction 0..1 (banks not listed are withdrawn), and optional `material_changes` [{row, col, material}] swapping assembly materials.
        """
        if not configurations:
            raise ToolError("give at least one configuration")
        if len(configurations) > MAX_CONFIGS:
            raise ToolError(f"at most {MAX_CONFIGS} configurations per call")
        labels = [c["label"] for c in configurations]
        if len(set(labels)) != len(labels):
            raise ToolError(f"labels must be unique, got {labels}")
        out, ref = [], None
        for cfg in configurations:
            changes = [(c["row"], c["col"], c["material"]) for c in cfg.get("material_changes", [])]
            try:
                xs, layout = core.build_xs(cfg.get("rods", {}), changes)
            except ValueError as e:
                raise ToolError(f"configuration {cfg['label']!r}: {e}") from e
            mask = core.fuel_mask(layout)
            t0 = time.perf_counter()
            sol = session.backend.predict(xs, core.pitch_cm, mask)
            r = Result(cfg, layout, mask, sol)
            session.results[cfg["label"]] = r
            ref = ref or r
            out.append({**_summary(r, ref), "compute_ms": round((time.perf_counter() - t0) * 1e3, 1)})
        return json.dumps({"backend": session.backend.name, "results": out})

    @beta_tool
    def get_map(
        label: str,
        quantity: Literal["power", "fast_flux", "thermal_flux", "material"],
        relative_to: str | None = None,
    ) -> str:
        """Return a full assembly map for an evaluated configuration, optionally as percent change against another evaluated configuration. Use for spatial questions (tilts, where flux is depressed) that the summaries do not answer.

        Args:
            label: Label of a configuration already run with evaluate_configurations.
            quantity: Which map. Flux is in the same relative units as power.
            relative_to: Optional second label; returns 100 * (label - relative_to) / relative_to over fuel assemblies.
        """
        for lab in (label, relative_to):
            if lab is not None and lab not in session.results:
                raise ToolError(f"no results for {lab!r}; evaluated: {sorted(session.results)}")
        r = session.results[label]
        if quantity == "material":
            return json.dumps({"label": label, "map": r.layout})
        pick = {"power": lambda s: s.power, "fast_flux": lambda s: s.flux[0], "thermal_flux": lambda s: s.flux[1]}[quantity]
        a = pick(r.sol)
        if relative_to is None:
            return json.dumps({"label": label, "quantity": quantity, "units": "relative (mean fuel power = 1)",
                               "map": _grid(np.where(r.fuel_mask | (quantity != "power"), a, np.nan), 4)})
        o = session.results[relative_to]
        both = r.fuel_mask & o.fuel_mask
        rel = np.full_like(a, np.nan)
        rel[both] = (a[both] - pick(o.sol)[both]) / pick(o.sol)[both] * 100
        return json.dumps({"label": label, "relative_to": relative_to, "quantity": quantity,
                           "units": "percent change, fuel assemblies only", "map": _grid(rel, 2)})

    return [describe_core, evaluate_configurations, get_map]
