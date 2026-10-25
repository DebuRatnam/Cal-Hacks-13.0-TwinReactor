"""Surrogate + reference solver behind the UI, on the training dataset's core.

The UI edits the 9x9 quarter-core assembly layout of the v3 dataset (NRC/OECD HELIOS UO2 tables):
six fuel types (4.2 / 4.5 wt% x 3 burnup states), fixed reflector and void positions, eight rod
banks, and soluble boron. Inputs are expanded to the 68x68 mesh with the same code that generated
the training data (`datagen.tabulated_cases.TabulatedCases` + `resolve_xs`), so the model sees
exactly what it was trained on.

Units: assembly maps are means over the mesh cells inside each assembly. Power is normalized to
mean 1 over fuel cells (dataset convention). Flux keeps the dataset normalization,
integral of (phi1 + phi2) dA = 1 over the quarter core, in cm^-2.
"""

from __future__ import annotations

import json
import time
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np

from datagen.physics_contract import resolve_xs
from datagen.tabulated_cases import TabulatedCases
from model.train import GridPredictor

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "fno_softplus.pt"
EVAL_REPORT = ROOT / "checkpoints" / "fno_softplus_test_eval.json"

# Only the geometry/library parts of the generation config matter here; the sampling fields
# are required by the constructor but unused (layouts come from the UI).
GEN_CONFIG = {
    "n_samples": 1,
    "seed": 20261029,
    "burnup_states_gwd_t": [0.15, 17.5, 37.5],
    "boron_states_ppm": [0, 500, 1000, 1500, 2000],
    "age_mix_concentration": 0.3,
}
VOID = -1


class LayoutError(ValueError):
    """Input the model cannot represent (shown to the user as a 422)."""


class CoreService:
    def __init__(self, checkpoint: Path = CHECKPOINT, device: str = "cpu"):
        self.cases = TabulatedCases(GEN_CONFIG)
        self.checkpoint = checkpoint
        self.device = device
        meta = self.predictor.saved["metadata"]
        if tuple(meta["grid_shape"]) != (68, 68) or not np.isclose(meta["pitch_cm"], self.cases.h):
            raise RuntimeError("Checkpoint mesh does not match the dataset geometry")
        self.meta = meta
        tc = self.cases
        self.n_fuel_types = len(tc.states)
        self.reflector_id = self.n_fuel_types
        # Per-assembly site type: "fuel", "reflector" or "void".
        self.site = np.where(tc.base < 0, "void", np.where(tc.fuel, "fuel", "reflector"))
        self.H, self.W = tc.base.shape

    @cached_property
    def predictor(self) -> GridPredictor:
        return GridPredictor(self.checkpoint, self.device)

    # ------------------------------------------------------------------ description
    def default_layout(self) -> list[list[int]]:
        """Placeholder loading pattern: fresh 4.5 wt% / once-burned 4.2 wt% checkerboard,
        twice-burned 4.2 wt% on the periphery (fuel next to reflector)."""
        tc = self.cases
        ids = tc.base.copy()
        fresh45, burned42, twice42 = 3, 1, 2
        for i, j in np.argwhere(tc.fuel):
            nbrs = [(i + di, j + dj) for di, dj in ((1, 0), (0, 1))]
            edge = any(a < self.H and b < self.W and self.site[a, b] != "fuel" for a, b in nbrs)
            ids[i, j] = twice42 if edge else (fresh45 if (i + j) % 2 == 0 else burned42)
        return ids.tolist()

    def describe(self) -> dict:
        tc = self.cases
        report = json.loads(EVAL_REPORT.read_text()) if EVAL_REPORT.exists() else {}
        materials = [
            {"id": n, "name": f"UO2 {e} wt%, {b} GWd/t", "enrichment_wt": e, "burnup_gwd_t": b, "fuel": True}
            for n, (e, b) in enumerate(tc.states)
        ] + [{"id": self.reflector_id, "name": "Reflector", "fuel": False}]
        banks = {str(int(b)): np.argwhere(tc.banks == b).tolist() for b in tc.bank_ids}
        return {
            "grid": [self.H, self.W],
            "site": self.site.tolist(),
            "materials": materials,
            "banks": banks,
            "default_layout": self.default_layout(),
            "default_boron_ppm": 1000.0,
            "assembly_pitch_cm": self.meta["assembly_pitch_cm"],
            "mesh": {
                "shape": [68, 68],
                "cell_cm": self.meta["pitch_cm"],
                "cells_per_assembly": 8,
                # Assembly row/column index of each mesh row/column (same for both axes).
                "blocks": tc.blocks.tolist(),
            },
            "geometry_note": "Row 0 / column 0 are half assemblies cut by the symmetry planes "
            "(reflective). Vacuum boundary on the outer (east/north and stepped) edges.",
            "training_ranges": {
                "boron_ppm": GEN_CONFIG["boron_states_ppm"],
                "rod_insertion": [0, 1],
                "temperatures": "fixed: fuel 900 K, coolant 580 K",
            },
            "model": {
                "name": "FNO (PhysicsNeMo), checkpoint fno_softplus",
                "dataset": self.meta["version"],
                "xs_source": self.meta["xs_source"],
                "test_k_mae_pcm": report.get("k_mae_pcm"),
                "test_k_p95_pcm": report.get("k_p95_abs_pcm"),
                "test_power_rel_l2": report.get("power_relative_l2_mean"),
                "test_samples": report.get("samples"),
            },
            "units": {
                "power": "relative; mean 1 over fuel cells; assembly value = mean over its mesh cells",
                "flux": "cm^-2; integral of (phi1 + phi2) dA = 1 over the quarter core; assembly mean",
                "k_eff": "absolute",
            },
            "scope": self.meta["scope"],
        }

    # ------------------------------------------------------------------ inputs
    def build(self, layout, rods: dict[str, float], boron_ppm: float):
        """Validate UI input and expand it to the solver/model case. Returns (case, warnings)."""
        tc = self.cases
        ids = np.asarray(layout)
        if ids.shape != (self.H, self.W) or not np.issubdtype(ids.dtype, np.integer):
            raise LayoutError(f"layout must be a {self.H}x{self.W} integer grid")
        for (i, j), kind in np.ndenumerate(self.site):
            v = int(ids[i, j])
            if kind == "void" and v != VOID:
                raise LayoutError(f"({i}, {j}) is outside the core; it must stay {VOID}")
            if kind == "reflector" and v != self.reflector_id:
                raise LayoutError(
                    f"({i}, {j}) is a reflector position; the model was only trained with the "
                    "fixed core boundary"
                )
            if kind == "fuel" and not 0 <= v < self.n_fuel_types:
                raise LayoutError(f"({i}, {j}) is a fuel position; use a fuel material id 0-5")
        if not 0 <= boron_ppm <= 2000:
            raise LayoutError("Soluble boron must be within the tabulated 0-2000 ppm range")

        warnings = []
        rod = np.zeros((self.H, self.W))
        for bank, frac in rods.items():
            if not bank.isdigit() or int(bank) not in tc.bank_ids:
                raise LayoutError(f"unknown rod bank {bank!r}; banks are {tc.bank_ids.tolist()}")
            if not 0 <= frac <= 1:
                raise LayoutError(f"bank {bank} insertion {frac} outside [0, 1]")
            rod[tc.banks == int(bank)] = frac
            if 0 < frac < 1:
                warnings.append(
                    f"Bank {bank} at {frac:.0%}: training used only fully withdrawn or fully "
                    "inserted banks. Partial insertion is a linear XS blend and is extrapolation."
                )
        if boron_ppm not in GEN_CONFIG["boron_states_ppm"]:
            warnings.append(
                f"Boron {boron_ppm:g} ppm lies between the sampled training states "
                f"{GEN_CONFIG['boron_states_ppm']} ppm (XS interpolated)."
            )
        case = tc.explicit(ids.astype(int), rod, float(boron_ppm))
        mask = case["material_map"] >= 0
        xs = resolve_xs(
            case["material_map"], case["xs_table"], case["rod_state"], case["rodded_xs_table"], mask
        )
        fuel = mask & np.any(xs[4:6] > 0, axis=0)
        return case, xs, mask, fuel, warnings

    # ------------------------------------------------------------------ outputs
    def _assembly_mean(self, field: np.ndarray, cells: np.ndarray) -> list[list[float | None]]:
        B = self.cases.blocks
        out = []
        for a in range(self.H):
            row = []
            for b in range(self.W):
                sel = np.ix_(B == a, B == b)
                m = cells[sel]
                row.append(round(float(field[sel][m].mean()), 6) if m.any() else None)
            out.append(row)
        return out

    @staticmethod
    def _cells(field: np.ndarray, keep: np.ndarray) -> list[list[float | None]]:
        return [
            [round(float(v), 5) if k else None for v, k in zip(row, krow)]
            for row, krow in zip(field, keep)
        ]

    def _summary(self, k_eff, flux, power, mask, fuel, source, seconds, warnings, extra=None):
        p_asm = self._assembly_mean(power, fuel)
        arr = np.array([[np.nan if v is None else v for v in row] for row in p_asm])
        peak_at = [int(v) for v in np.unravel_index(np.nanargmax(arr), arr.shape)]
        return {
            "source": source,
            "k_eff": round(float(k_eff), 6),
            "power": p_asm,
            # x1e5 so typical values are O(1); the unit string says so.
            "fast": self._assembly_mean(flux[0] * 1e5, mask),
            "thermal": self._assembly_mean(flux[1] * 1e5, mask),
            "flux_unit": "1e-5 cm^-2 (integral of phi1+phi2 over quarter core = 1)",
            "assembly_peaking": round(float(np.nanmax(arr)), 4),
            "assembly_peak_row_col": peak_at,
            "node_peaking": round(float(power[fuel].max()), 4),
            # Full-resolution 68x68 maps, same units (flux x1e5); None outside fuel (power) or
            # outside the core (flux). Row 0 = south, column 0 = west, like the assembly maps.
            "mesh": {
                "power": self._cells(power, fuel),
                "fast": self._cells(flux[0] * 1e5, mask),
                "thermal": self._cells(flux[1] * 1e5, mask),
            },
            "compute_ms": round(seconds * 1e3, 1),
            "warnings": warnings,
            **(extra or {}),
        }

    def predict(self, layout, rods, boron_ppm) -> dict:
        _, xs, mask, fuel, warnings = self.build(layout, rods, boron_ppm)
        t0 = time.perf_counter()
        pred = self.predictor.predict(
            {"x": xs[None].astype(np.float32), "mask": mask[None], "fuel_mask": fuel[None]}
        )
        dt = time.perf_counter() - t0
        # The network does not enforce the dataset normalizations exactly (raw power mean over
        # fuel ~0.98-1.02). They are definitions, not physics, so rescale and report the raw
        # values so the correction stays visible.
        flux, power = pred["flux"][0], pred["power"][0]
        # Physics checks (CLAUDE.md): flux >= 0 and power >= 0 in fuel. The checkpoint applies
        # softplus to flux only, so power can dip slightly below zero in near-zero-power edge
        # cells. Clip to 0 before normalizing (approved by EECS, 2026-10-25); the raw minimum and
        # count stay in `raw_normalization`. Proper fix: softplus on the power head + retrain.
        warnings = list(warnings)
        p_min, neg = float(power[fuel].min()), int((power[fuel] < 0).sum())
        power = np.maximum(power, 0.0)
        if (flux[:, mask] < 0).any():
            warnings.append("Physics check: negative predicted flux in the core.")
        p_mean = float(power[fuel].mean())
        f_int = float(flux[:, mask].sum() * self.cases.h**2)
        return self._summary(
            pred["k_eff"][0], flux / f_int, power / p_mean, mask, fuel,
            "surrogate: " + self.describe_model_short(), dt, warnings,
            {
                "raw_normalization": {
                    "power_mean_over_fuel": round(p_mean, 4),
                    "flux_integral": round(f_int, 4),
                    "power_min_before_clip": round(p_min, 4),
                    "negative_power_cells_clipped": neg,
                }
            },
        )

    def solve(self, layout, rods, boron_ppm) -> dict:
        case, _, mask, fuel, warnings = self.build(layout, rods, boron_ppm)
        t0 = time.perf_counter()
        sample, _ = self.cases.solve(case)
        dt = time.perf_counter() - t0
        return self._summary(
            sample.k_eff, sample.flux, sample.power, mask, fuel,
            f"reference: ndiffusion {self.meta['solver_version']} 2-group diffusion, 68x68 mesh",
            dt, warnings,
            {"solver": {k: sample.solver_info[k] for k in ("iterations", "balance_residual")}},
        )

    def describe_model_short(self) -> str:
        return f"FNO fno_softplus ({self.meta['version']})"


@lru_cache(maxsize=1)
def get_service() -> CoreService:
    """Process-wide service shared by the API and the agent (checkpoint loaded once)."""
    return CoreService()
