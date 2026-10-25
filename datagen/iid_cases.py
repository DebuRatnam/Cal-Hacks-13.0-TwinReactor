"""One reproducible distribution of fuel layouts and binary rod configurations.

Preparing or previewing cases never calls a physics solver. The catalog is the
pinned IAEA benchmark; rod-bearing blocks retain fuel 2 because only that fuel
has a matched rodded endpoint in the supplied benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from datagen.physics_contract import XS_FIELDS, resolve_xs


def configuration_hash(case):
    return hashlib.sha256(
        np.asarray(case["material_map"], dtype="<i8").tobytes()
        + np.asarray(case["rod_state"], dtype="<f8").tobytes()
    ).hexdigest()


def grouped_splits(hashes, seed):
    """Preserve IID draws, including repeats, but never split identical inputs."""
    splits = {"train": [], "val": [], "test": []}
    for i, value in enumerate(hashes):
        score = (
            int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big") / 2**64
        )
        key = "train" if score < 0.8 else ("val" if score < 0.9 else "test")
        splits[key].append(i)
    return splits


class IIDCases:
    def __init__(self, config: dict, config_dir: Path):
        spec = config["sampling"]
        template_path = config_dir / spec["template"]
        template_bytes = template_path.read_bytes()
        template = json.loads(template_bytes)
        self.seed = int(config["seed"])
        self.n = int(config["n_samples"])
        if self.seed < 0 or self.n < 1:
            raise ValueError("Use a nonnegative seed and positive n_samples")
        self.geometry = spec.get("geometry", "quarter")
        if self.geometry not in ("quarter", "full"):
            raise ValueError("geometry must be quarter or full")
        if spec.get("distribution") != "uniform_count_then_uniform_subset":
            raise ValueError("Specify the documented common sampling distribution")
        # Sample one material per assembly, not independent material pixels.
        h = float(template["pitch_cm"])
        if not np.isclose(h, 2.5):
            raise ValueError("This pinned IAEA template uses 2.5 cm cells")
        edges = np.array([0, 10, 30, 50, 70, 90, 110, 130, 150, 170])
        centers = ((edges[:-1] + edges[1:]) / (2 * h)).astype(int)
        reference = template["cases"][0]
        fine_ids = np.asarray(reference["material_map"], dtype=int)
        fine_rod = np.asarray(reference["rod_state"], dtype=float)
        base = fine_ids[np.ix_(centers, centers)]
        eligible = fine_rod[np.ix_(centers, centers)] > 0
        if self.geometry == "full":
            # Merge the two 10 cm half-blocks on each symmetry axis into one
            # 20 cm central assembly, giving a 17x17 full-core assembly grid.
            mirror = np.r_[np.arange(8, 0, -1), np.arange(9)]
            base = base[np.ix_(mirror, mirror)]
            eligible = eligible[np.ix_(mirror, mirror)]
            self.pixel_blocks = np.arange(136) // 8
        else:
            self.pixel_blocks = np.searchsorted(edges, (np.arange(68) + 0.5) * h) - 1
        self.base = base
        self.eligible = eligible
        if not np.isin(base, [-1, 0, 1, 3]).all() or not eligible.any():
            raise ValueError("Unexpected IAEA template materials/rod locations")
        self.free_fuel = np.flatnonzero(np.isin(base, [0, 1]) & ~eligible)
        self.rod_sites = np.flatnonzero(eligible)
        self.table = np.asarray(template["xs_table"], dtype=float)
        self.rodded = np.asarray(template["rodded_xs_table"], dtype=float)
        if tuple(template["xs_fields"]) != XS_FIELDS or np.array_equal(
            self.table[1], self.rodded[1]
        ):
            raise ValueError("A matched fuel-2 rodded endpoint is required")
        # Preserve all shared physics settings; the sampling changes only the
        # eligible material choices and binary rod patterns.
        self.config = {key: value for key, value in template.items() if key != "cases"}
        self.config.update(config)
        self.config["sampling"] = dict(spec)
        self.config["sampling_template_sha256"] = hashlib.sha256(template_bytes).hexdigest()
        self.config["xs_source"] = (
            "Pinned IAEA two-group benchmark tables from the template; no independent property perturbations"
        )
        self.config["scope"] = (
            "IID IAEA-derived 2D rod/fuel configuration study. Fixed eligible rod sites; "
            "binary independently selectable rod states, not a plant operating-bank model. "
            "Two benchmark unrodded fuel types; rod-capable blocks fixed to fuel 2. "
            "No temperature, added absorption, boron or burnup variation. "
            + (
                "Quarter core imposes mirrored configurations."
                if self.geometry == "quarter"
                else "Full 2D core permits independent asymmetric layouts and rod states."
            )
        )
        if self.geometry == "full":
            self.config["bc"] = {side: "vacuum" for side in template["bc"]}

    def __len__(self):
        return self.n

    def block_case(self, i: int):
        if not 0 <= i < self.n:
            raise IndexError(i)
        # Per-index seeds give reproducible random draws and a stable prefix
        # when n_samples changes, independent of job ordering or resumption.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, i]))
        ids = self.base.copy()
        ids.flat[self.free_fuel] = 0
        fuel_count = int(rng.integers(len(self.free_fuel) + 1))
        selected = rng.choice(self.free_fuel, size=fuel_count, replace=False)
        ids.flat[selected] = 1
        ids[self.eligible] = 1
        rod = np.zeros(ids.shape)
        rod_count = int(rng.integers(len(self.rod_sites) + 1))
        active = rng.choice(self.rod_sites, size=rod_count, replace=False)
        rod.flat[active] = 1
        return ids, rod

    def __getitem__(self, i: int):
        ids, rod = self.block_case(i)
        idx = np.ix_(self.pixel_blocks, self.pixel_blocks)
        return {"material_map": ids[idx], "rod_state": rod[idx]}

    def preview(self, n: int = 1000):
        count = min(n, self.n)
        rod_hist = np.zeros(len(self.rod_sites) + 1, int)
        fuel_counts, hashes = [], set()
        for i in range(count):
            ids, rod = self.block_case(i)
            rod_hist[int(rod.sum())] += 1
            fuel_counts.append(int(np.count_nonzero(ids.ravel()[self.free_fuel] == 1)))
            hashes.add(hashlib.sha256(ids.tobytes() + rod.tobytes()).hexdigest())
        # Validate resolved properties on one input without solving an equation.
        case = self[0]
        resolve_xs(case["material_map"], self.table, case["rod_state"], self.rodded)
        return {
            "planned_samples": self.n,
            "previewed_samples": count,
            "unique_preview_inputs": len(hashes),
            "geometry": self.geometry,
            "grid_shape": list(case["material_map"].shape),
            "free_fuel_assemblies": len(self.free_fuel),
            "fixed_rod_capable_locations": len(self.rod_sites),
            "possible_rod_patterns": 2 ** len(self.rod_sites),
            "possible_fuel_and_rod_patterns": str(2 ** (len(self.free_fuel) + len(self.rod_sites))),
            "preview_rod_count_histogram": rod_hist.tolist(),
            "preview_variable_fuel2_count_range": [min(fuel_counts), max(fuel_counts)],
            "sampling_law": "For every row: uniform fuel-2 count, uniform positions given count; independently uniform inserted-rod count, uniform eligible positions given count",
            "all_combinations_equally_likely": False,
            "scope": self.config["scope"],
            "physics_solves_performed": 0,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int)
    parser.add_argument("--geometry", choices=["quarter", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.n is not None:
        cfg["n_samples"] = args.n
    if args.geometry:
        cfg["sampling"]["geometry"] = args.geometry
        cfg["version"] += "-" + args.geometry
    cases = IIDCases(cfg, args.config.parent)
    args.out.mkdir(parents=True, exist_ok=True)
    report = cases.preview()
    (args.out / "sampling_plan.json").write_text(json.dumps(report, indent=2) + "\n")
    preview = [cases[i] for i in range(min(8, len(cases)))]
    np.savez_compressed(
        args.out / "input_preview.npz",
        material_map=np.stack([case["material_map"] for case in preview]),
        rod_state=np.stack([case["rod_state"] for case in preview]),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
