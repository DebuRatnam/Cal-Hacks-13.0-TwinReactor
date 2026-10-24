"""On-disk dataset format shared by datagen, model, and backend.

Layout of one dataset directory:
    samples.h5      datasets: material_map [N,H,W] int, k_eff [N], flux [N,G,H,W], power [N,H,W]
    metadata.json   DatasetMetadata
    splits.json     {"train": [...], "val": [...], "test": [...]} sample indices
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np


@dataclass
class DatasetMetadata:
    version: str
    solver: str  # "ndiffusion" | "openmc"
    solver_version: str
    xs_source: str
    n_groups: int
    grid_shape: tuple[int, int]
    pitch_cm: float
    seed: int
    git_commit: str = ""
    material_names: list[str] = field(default_factory=list)


@dataclass
class Sample:
    material_map: np.ndarray  # [H, W] int, index into material_names
    k_eff: float
    flux: np.ndarray  # [G, H, W]
    power: np.ndarray  # [H, W], mean 1 over fuel cells


def write_dataset(out_dir: Path, meta: DatasetMetadata, samples: list[Sample]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(asdict(meta), indent=2))
    with h5py.File(out_dir / "samples.h5", "w") as f:
        f["material_map"] = np.stack([s.material_map for s in samples])
        f["k_eff"] = np.array([s.k_eff for s in samples])
        f["flux"] = np.stack([s.flux for s in samples])
        f["power"] = np.stack([s.power for s in samples])


def read_metadata(data_dir: Path) -> DatasetMetadata:
    raw = json.loads((data_dir / "metadata.json").read_text())
    raw["grid_shape"] = tuple(raw["grid_shape"])
    return DatasetMetadata(**raw)


def make_splits(n: int, seed: int, frac: tuple[float, float] = (0.8, 0.1)) -> dict[str, list[int]]:
    idx = np.random.default_rng(seed).permutation(n)
    n_train, n_val = int(frac[0] * n), int(frac[1] * n)
    return {
        "train": idx[:n_train].tolist(),
        "val": idx[n_train : n_train + n_val].tolist(),
        "test": idx[n_train + n_val :].tolist(),
    }
