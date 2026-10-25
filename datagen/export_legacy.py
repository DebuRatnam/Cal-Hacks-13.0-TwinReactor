"""Lossless full-field export of the existing pilot/stress solves for the models.

This changes the file interface, not the scope of the old physics experiments.
Material properties are reconstructed from their saved explicit inputs and the
pinned IAEA tables, not estimated from flux or k_eff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import ndiffusion as nd
import numpy as np

from datagen.physics_contract import QUARTER_CORE_BC
from datagen.schema import DatasetMetadata, Sample, write_dataset


def maps_from_materials(mats, ids, mask):
    n = mats.n_mat
    table = np.column_stack(
        [
            np.asarray(mats.D).reshape(n, 2),
            np.asarray(mats.removal).reshape(n, 2),
            np.asarray(mats.nusigf).reshape(n, 2),
            np.asarray(mats.scatter).reshape(n, 2, 2)[:, 1, 0],
        ]
    )
    xs = np.zeros((7, *mask.shape))
    xs[:, mask] = table[ids[mask]].T
    return xs


def power_from_flux(flux, mats, physics_ids, mask):
    q = nd.power_density(flux[:, mask].T, mats, physics_ids[mask])
    nf = np.asarray(mats.nusigf).reshape(mats.n_mat, 2)
    fuel = np.any(nf[physics_ids[mask]] > 0, axis=1)
    result = np.zeros(mask.shape)
    result[mask] = q / q[fuel].mean()
    return result


def load_source(root, kind):
    manifest = json.loads((root / "data/manifest.json").read_text())
    if manifest["ndiffusion_version"] != nd.__version__ or manifest["schema"] != 1:
        raise ValueError("Legacy data must use the pinned schema/ndiffusion material definitions")
    with np.load(root / "data/dataset.npz", allow_pickle=False) as z:
        data = {name: z[name].copy() for name in z.files}
    if not np.all(data["converged"]):
        raise ValueError("Cannot export nonconverged reference cases")
    flux, mask = data["flux"], data["mask"]
    h = float(data["h_cm"])
    rows = np.array(nd.materials.IAEA.rows)
    cases = []
    families = []
    for i, phi in enumerate(flux):
        if kind == "pilot":
            table = np.vstack([rows, rows[2]])
            inner, outer, absorption = data["inputs"][i]
            table[2, 3] = rows[1, 3] + inner * (rows[2, 3] - rows[1, 3])
            table[4, 3] = rows[1, 3] + outer * (rows[2, 3] - rows[1, 3])
            table[[0, 1, 2, 4], 3] += absorption
            physics_ids = data["region_id"].copy()
            layout = physics_ids.copy()
            layout[(layout == 2) | (layout == 4)] = 1
            rod = np.zeros(mask.shape)
            rod[physics_ids == 2] = inner
            rod[physics_ids == 4] = outer
            family = "original_pilot"
        else:
            # Exact legacy IAEA block edges; first block is a 10 cm half block.
            edges = np.array([0, 10, 30, 50, 70, 90, 110, 130, 150, 170])
            by = np.searchsorted(edges, (np.arange(mask.shape[0]) + 0.5) * h, side="right") - 1
            bx = np.searchsorted(edges, (np.arange(mask.shape[1]) + 0.5) * h, side="right") - 1
            physics_ids = by[:, None] * 9 + bx[None, :]
            fuel = data["fuel_code"][i]
            strengths = data["rod_strength"][i]
            table = np.tile(rows[3], (81, 1))
            for code in (1, 2):
                table[fuel.ravel() == code] = rows[code - 1]
            active_fuel = np.isin(fuel.ravel(), [1, 2])
            table[active_fuel, 3] += (rows[2, 3] - rows[1, 3]) * strengths.ravel()[
                active_fuel
            ] + data["absorption_added"][i]
            layout = fuel[by[:, None], bx[None, :]] - 1
            rod = strengths[by[:, None], bx[None, :]]
            family = str(data["suite"][i])
        mats = nd.materials.two_group(table, axial_buckling=nd.materials.IAEA.axial_buckling)
        layout = layout.astype(np.int64)
        layout[~mask] = -1
        info = {
            name: np.asarray(data[name][i]).item()
            for name in ("converged", "iterations", "residual", "seconds")
        }
        balance = float(data["balance_residual"][i])
        info["balance_residual"] = balance if np.isfinite(balance) else None
        cases.append(
            Sample(
                layout,
                float(data["keff"][i]),
                phi,
                power_from_flux(phi, mats, physics_ids, mask),
                maps_from_materials(mats, physics_ids, mask),
                rod,
                mask.copy(),
                info,
            )
        )
        families.append(family)
    return cases, data["split"], families, h, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path)
    p.add_argument("--stress", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261024)
    args = p.parse_args()
    if args.pilot is None and args.stress is None:
        p.error("Provide --pilot and/or --stress")
    samples = []
    labels = []
    family_labels = []
    sources = []
    spacing = None
    for kind, root in (("pilot", args.pilot), ("stress", args.stress)):
        if root is None:
            continue
        cases, split, families, h, manifest = load_source(root, kind)
        if spacing is not None and h != spacing:
            raise ValueError("Mixed grid spacings are not allowed")
        spacing = h
        offset = len(samples)
        samples.extend(cases)
        labels.extend(split.tolist())
        family_labels.extend(families)
        sources.append(
            {
                "kind": kind,
                "start": offset,
                "count": len(cases),
                "manifest": manifest,
                "dataset_sha256": hashlib.sha256(
                    (root / "data/dataset.npz").read_bytes()
                ).hexdigest(),
            }
        )
    labels = np.asarray(labels)
    families = np.asarray(family_labels)
    splits = {"train": [], "val": [], "test": np.flatnonzero(labels != "train").tolist()}
    rng = np.random.default_rng(args.seed)
    # Preserve all old held-outs, including pilot corners. Validation is carved
    # only from the old training rows, independently in each source family.
    for family in np.unique(families):
        idx = np.flatnonzero((families == family) & (labels == "train"))
        idx = rng.permutation(idx)
        nv = max(1, round(0.1 * len(idx))) if len(idx) > 1 else 0
        splits["val"].extend(idx[:nv].tolist())
        splits["train"].extend(idx[nv:].tolist())
    assert sorted(i for indices in splits.values() for i in indices) == list(range(len(samples)))
    meta = DatasetMetadata(
        version="v1-legacy-full-fields",
        solver="ndiffusion",
        solver_version=nd.__version__,
        xs_source="Pinned IAEA tables plus the saved explicit legacy input perturbations",
        n_groups=2,
        grid_shape=samples[0].material_map.shape,
        pitch_cm=spacing,
        seed=args.seed,
        material_names=["fuel_1", "fuel_2", "rodded_fuel_2", "reflector"],
        bc=dict(QUARTER_CORE_BC),
        boundary_gamma=0.4692,
        flux_normalization="unit_integral_total_2group",
        assembly_pitch_cm=20.0,
        scope="Legacy quarter-core pilot/stress experiments only; this export does not sample independently varied general material properties or unrestricted full-core configurations",
    )
    write_dataset(args.out, meta, samples)
    (args.out / "splits.json").write_text(json.dumps(splits, indent=2))
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "family": family_labels,
                "original_split": labels.tolist(),
                "test_policy": "All original test and corner cases remain held out; val drawn from original train only",
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "samples": len(samples),
                "shape": list(samples[0].flux.shape),
                "input_channels": 7,
                "splits": {k: len(v) for k, v in splits.items()},
                "boundary_gamma": 0.4692,
                "cell_width_cm": spacing,
                "scope": meta.scope,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
