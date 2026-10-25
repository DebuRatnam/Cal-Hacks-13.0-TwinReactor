"""Bulk training data generation with ndiffusion.

Usage:
    python -m datagen.ndiffusion_gen --config datagen/configs/iaea_explicit_examples.json --out data/examples
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from time import perf_counter

import ndiffusion as nd
import numpy as np
import yaml

from datagen.physics_contract import (
    MARSHAK,
    QUARTER_CORE_BC,
    XS_FIELDS,
    compact_xs,
    resolve_xs,
)
from datagen.schema import DatasetMetadata, Sample, make_splits, write_dataset


def materials_from_table(table):
    """Use resolved removal directly; never add outgoing scatter twice."""
    table = np.asarray(table, dtype=float)
    m = nd.Materials()
    m.n_mat, m.n_groups = len(table), 2
    m.D = table[:, :2].ravel().tolist()
    m.removal = table[:, 2:4].ravel().tolist()
    m.nusigf = table[:, 4:6].ravel().tolist()
    m.scatter = (
        np.stack(
            [np.zeros(len(table)), np.zeros(len(table)), table[:, 6], np.zeros(len(table))], axis=1
        )
        .ravel()
        .tolist()
    )
    m.chi = np.tile([1.0, 0.0], len(table)).tolist()
    return m


def build_mesh(ids, mask, h_cm, table, bc, gamma):
    """Uniform, cell-centered quad mesh with the actual stepped outer boundary."""
    if h_cm <= 0 or gamma < 0 or set(bc) != set(QUARTER_CORE_BC):
        raise ValueError("Invalid spacing or boundary conditions")
    if any(v not in ("vacuum", "reflective") for v in bc.values()):
        raise ValueError("Only reflective/vacuum boundaries are supported by these models")
    H, W = ids.shape
    vertices, vx, vy = {}, [], []

    def vertex(x, y):
        key = (x, y)
        if key not in vertices:
            vertices[key] = len(vx)
            vx.append(x * h_cm)
            vy.append(y * h_cm)
        return vertices[key]

    cv, offsets, mids, edges = [], [0], [], {}
    cells = np.argwhere(mask)
    for y, x in cells:
        v = [vertex(x, y), vertex(x + 1, y), vertex(x + 1, y + 1), vertex(x, y + 1)]
        cv.extend(v)
        offsets.append(len(cv))
        mids.append(int(ids[y, x]))
        for a, b in zip(v, v[1:] + v[:1]):
            key = (min(a, b), max(a, b))
            if key in edges:
                edges[key] = None
            else:
                edges[key] = int(ids[y, x])
    a0, a1, tags = [], [], []
    for (a, b), mid in edges.items():
        if mid is None:
            continue
        side = None
        if vx[a] == vx[b] == 0:
            side = "west"
        elif vx[a] == vx[b] == W * h_cm:
            side = "east"
        elif vy[a] == vy[b] == 0:
            side = "south"
        elif vy[a] == vy[b] == H * h_cm:
            side = "north"
        reflective = side is not None and bc[side] == "reflective"
        a0.append(a)
        a1.append(b)
        tags.append(0 if reflective else 1 + mid)
    mesh = nd.UnstructuredMesh2D()
    mesh.vx, mesh.vy = vx, vy
    mesh.cell_vertices, mesh.cell_offsets, mesh.material_id = cv, offsets, mids
    mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag = a0, a1, tags
    conditions = [nd.BoundaryCondition(A=0, B=1), nd.BoundaryCondition(A=0, B=1)]
    for row in table:
        # Local D supports configurable properties even at the exterior.
        conditions.extend(
            [
                nd.BoundaryCondition(A=gamma, B=float(row[0])),
                nd.BoundaryCondition(A=gamma, B=float(row[1])),
            ]
        )
    return mesh, conditions, cells


def solve_case(material_map, params: dict) -> Sample:
    """Build nd.Materials + nd.KEigenSolver for one layout and return the solution.

    All layout IDs, base properties, rod endpoints and rod state are inputs.
    Grid spacing is a numerical cell width, not necessarily assembly pitch.
    """
    start = perf_counter()
    ids = np.asarray(material_map)
    if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("material_map must be a 2-D integer array")
    ids = ids.astype(np.int64)
    mask = ids >= 0
    rod = np.asarray(params.get("rod_state", np.zeros(ids.shape)), dtype=float)
    xs = resolve_xs(ids, params["xs_table"], rod, params.get("rodded_xs_table"), mask)
    physics_ids, table, void_id = compact_xs(xs, mask)
    # The unused void row is padding, not a zero-D physical material.
    table = table[:void_id]
    h = float(params["pitch_cm"])
    bc = dict(params.get("bc", QUARTER_CORE_BC))
    gamma = float(params.get("boundary_gamma", MARSHAK))
    mesh, conditions, cells = build_mesh(physics_ids, mask, h, table, bc, gamma)
    mats = materials_from_table(table)
    with warnings.catch_warnings():
        warnings.simplefilter("error", nd.ConvergenceWarning)
        r = nd.KEigenSolverUnstructured2D(
            mats, mesh, conditions, epsilon=1e-8, max_outer=4000, max_inner=300, use_cg=True
        ).solve()
    raw = np.asarray(r.flux).reshape(-1, 2)
    if not r.converged or not np.isfinite(raw).all() or (raw < 0).any():
        raise RuntimeError("Rejected a failed or invalid reference solve")
    raw = raw / (raw.sum() * h * h)
    flux = np.zeros((2, *ids.shape))
    flux[:, cells[:, 0], cells[:, 1]] = raw.T
    power_cells = nd.power_density(raw, mats, mesh.material_id)
    fuel_cells = np.any(table[np.asarray(mesh.material_id), 4:6] > 0, axis=1)
    power = np.zeros(ids.shape)
    power[cells[:, 0], cells[:, 1]] = power_cells / np.mean(power_cells[fuel_cells])
    op = nd.unstructured_operator(mats, mesh, conditions)
    source = op.production @ raw.ravel() / r.keff
    balance = float(np.linalg.norm(op.loss @ raw.ravel() - source) / np.linalg.norm(source))
    if balance >= 1e-6:
        raise RuntimeError(f"Discrete balance residual {balance} exceeds tolerance")
    return Sample(
        ids,
        float(r.keff),
        flux,
        power,
        xs,
        rod,
        mask,
        {
            "converged": True,
            "iterations": int(r.iterations),
            "residual": float(r.residual),
            "balance_residual": balance,
            "seconds_total": perf_counter() - start,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if "cases" in cfg and "sampling" in cfg:
        raise ValueError("Provide either explicit cases or one sampling distribution")
    if "sampling" in cfg:
        from datagen.iid_cases import IIDCases

        cases = IIDCases(cfg, args.config.parent)
        cfg = cases.config
    elif "cases" in cfg:
        cases = cfg["cases"]
    else:
        raise ValueError(
            "Provide explicit cases with material_map and configurable property tables; do not infer XS from uncalibrated enrichment/boron fields"
        )
    if tuple(cfg.get("xs_fields", XS_FIELDS)) != XS_FIELDS:
        raise ValueError("Configuration XS columns do not match the model contract")
    args.out.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()
        + Path(__file__).read_bytes()
        + Path(__file__).with_name("physics_contract.py").read_bytes()
        + Path(__file__).with_name("iid_cases.py").read_bytes()
        + nd.__version__.encode()
    ).hexdigest()
    marker = args.out / "generation.json"
    if marker.exists() and json.loads(marker.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Existing output has a different generation config; use a new version")
    if (args.out / "samples.h5").exists():
        if not marker.exists():
            raise FileExistsError("Unrecognized dataset already exists")
        if not all(
            (args.out / name).exists() for name in ("metadata.json", "splits.json", "inputs.json")
        ):
            raise ValueError("Dataset is incomplete; preserve it before starting a fresh output")
        print("Matching dataset already complete")
        return
    marker.write_text(
        json.dumps({"fingerprint": fingerprint, "solver_version": nd.__version__}, indent=2)
    )
    shards = args.out / "shards"
    shards.mkdir(exist_ok=True)
    samples = []
    input_hashes = []
    for i, case in enumerate(cases):
        if "sampling" in cfg:
            from datagen.iid_cases import configuration_hash

            input_hashes.append(configuration_hash(case))
        shard = shards / f"{i:06d}.npz"
        info = shard.with_suffix(".json")
        if shard.exists() and info.exists():
            with np.load(shard, allow_pickle=False) as z:
                sample = Sample(
                    **{k: z[k].copy() for k in z.files}, solver_info=json.loads(info.read_text())
                )
        else:
            params = {
                k: cfg[k]
                for k in ("xs_table", "rodded_xs_table", "pitch_cm", "bc", "boundary_gamma")
                if k in cfg
            }
            params.update(
                {k: case[k] for k in ("xs_table", "rodded_xs_table", "rod_state") if k in case}
            )
            sample = solve_case(case["material_map"], params)
            temp = shard.with_suffix(".tmp.npz")
            np.savez_compressed(
                temp,
                **{
                    k: getattr(sample, k)
                    for k in ("material_map", "k_eff", "flux", "power", "xs", "rod_state", "mask")
                },
            )
            temp.replace(shard)
            info.write_text(json.dumps(sample.solver_info, indent=2))
        samples.append(sample)
        print(f"{i + 1}/{len(cases)}: k_eff={sample.k_eff:.8f}", flush=True)
    meta = DatasetMetadata(
        version=cfg["version"],
        solver="ndiffusion",
        solver_version=nd.__version__,
        xs_source=cfg["xs_source"],
        n_groups=2,
        grid_shape=samples[0].material_map.shape,
        pitch_cm=cfg["pitch_cm"],
        seed=cfg["seed"],
        bc=dict(cfg.get("bc", QUARTER_CORE_BC)),
        boundary_gamma=cfg.get("boundary_gamma", MARSHAK),
        flux_normalization="unit_integral_total_2group",
        scope=cfg.get(
            "scope", "Explicit supplied layouts, material properties and rod configurations"
        ),
        git_commit=cfg.get("git_commit", ""),
        material_names=cfg.get("material_names", []),
        assembly_pitch_cm=cfg.get("assembly_pitch_cm"),
    )
    (args.out / "inputs.json").write_text(json.dumps(cfg, indent=2))
    if "sampling" in cfg:
        from datagen.iid_cases import grouped_splits

        splits = grouped_splits(input_hashes, cfg["seed"])
        (args.out / "input_hashes.json").write_text(json.dumps(input_hashes, indent=2))
    else:
        splits = make_splits(len(samples), cfg["seed"])
    (args.out / "splits.json").write_text(json.dumps(splits, indent=2))
    write_dataset(args.out, meta, samples)


if __name__ == "__main__":
    main()
