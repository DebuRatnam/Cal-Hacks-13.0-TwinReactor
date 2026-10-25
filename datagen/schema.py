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

from datagen.physics_contract import QUARTER_CORE_BC, XS_FIELDS, validate_xs


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
    xs_fields: list[str] = field(default_factory=lambda: list(XS_FIELDS))
    bc: dict[str, str] = field(default_factory=lambda: dict(QUARTER_CORE_BC))
    boundary_gamma: float = 0.5
    flux_normalization: str = "unspecified"
    power_normalization: str = "mean_1_over_fuel"
    physics: str = "2d_2group_downscatter_chi_fast"
    scope: str = ""
    assembly_pitch_cm: float | None = None
    grid_orientation: str = "row0_south_col0_west"


@dataclass
class Sample:
    material_map: np.ndarray  # [H, W] int, index into material_names
    k_eff: float
    flux: np.ndarray  # [G, H, W]
    power: np.ndarray  # [H, W], mean 1 over fuel cells
    xs: np.ndarray | None = None  # [7,H,W], resolved solver coefficients, physical units
    rod_state: np.ndarray | None = None  # [H,W], explicit effective rod state
    mask: np.ndarray | None = None  # [H,W], includes fuel and reflector; excludes padding
    solver_info: dict | None = None


def write_dataset(out_dir: Path, meta: DatasetMetadata, samples: list[Sample]) -> None:
    if not samples or meta.n_groups != 2 or meta.physics != "2d_2group_downscatter_chi_fast":
        raise ValueError("Expected nonempty, two-group downscatter data for the current models")
    if (
        meta.xs_fields != list(XS_FIELDS)
        or meta.pitch_cm <= 0
        or meta.boundary_gamma < 0
        or meta.grid_orientation != "row0_south_col0_west"
    ):
        raise ValueError("Invalid XS order, grid spacing or boundary coefficient")
    if set(meta.bc) != set(QUARTER_CORE_BC) or any(
        v not in ("vacuum", "reflective") for v in meta.bc.values()
    ):
        raise ValueError("Specify reflective/vacuum conditions on all four outer sides")
    optional = ("xs", "rod_state", "mask", "solver_info")
    for name in optional:
        if any(getattr(s, name) is not None for s in samples) and not all(
            getattr(s, name) is not None for s in samples
        ):
            raise ValueError(f"All samples must consistently provide {name}")
    masks = []
    for s in samples:
        if np.asarray(s.material_map).shape != meta.grid_shape or not np.issubdtype(
            np.asarray(s.material_map).dtype, np.integer
        ):
            raise ValueError("material_map shape/dtype does not match metadata")
        mask = np.asarray(s.mask, bool) if s.mask is not None else np.asarray(s.material_map) >= 0
        if mask.shape != meta.grid_shape or not mask.any() or (s.material_map[mask] < 0).any():
            raise ValueError("Invalid domain mask or active material ID")
        if (
            np.asarray(s.flux).shape != (2, *meta.grid_shape)
            or np.asarray(s.power).shape != meta.grid_shape
        ):
            raise ValueError("Target shapes do not match metadata")
        if not np.isfinite(s.k_eff) or s.k_eff <= 0:
            raise ValueError("k_eff must be finite and positive")
        if (
            not np.isfinite(s.flux).all()
            or not np.isfinite(s.power).all()
            or (s.flux < 0).any()
            or (s.power < 0).any()
        ):
            raise ValueError("Targets must be finite and nonnegative")
        if np.any(s.flux[:, ~mask] != 0) or np.any(s.power[~mask] != 0):
            raise ValueError("Targets outside the core must be zero padding")
        if s.xs is not None:
            validate_xs(s.xs, mask)
            fuel = np.any(s.xs[4:6] > 0, axis=0) & mask
            if not np.allclose(np.mean(s.power[fuel]), 1, rtol=1e-5, atol=1e-6) or np.any(
                s.power[~fuel] != 0
            ):
                raise ValueError("Power must be zero outside fuel and have mean 1 over fuel")
        if s.rod_state is not None:
            rod = np.asarray(s.rod_state)
            if (
                rod.shape != meta.grid_shape
                or not np.isfinite(rod).all()
                or ((rod < 0) | (rod > 1)).any()
                or np.any(rod[~mask] != 0)
            ):
                raise ValueError("Invalid rod_state")
        if meta.flux_normalization == "unit_integral_total_2group" and not np.isclose(
            np.sum(s.flux) * meta.pitch_cm**2, 1, rtol=1e-5
        ):
            raise ValueError("Flux does not obey the recorded common two-group normalization")
        masks.append(mask)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "samples.h5").exists():
        raise FileExistsError("Dataset already exists; use a new version/output directory")
    with h5py.File(out_dir / "samples.tmp.h5", "w") as f:
        f.create_dataset(
            "material_map",
            data=np.stack([s.material_map for s in samples]),
            chunks=(1, *meta.grid_shape),
            compression="gzip",
            shuffle=True,
        )
        f["k_eff"] = np.array([s.k_eff for s in samples])
        f.create_dataset(
            "flux",
            data=np.stack([s.flux for s in samples]),
            chunks=(1, 2, *meta.grid_shape),
            compression="gzip",
            shuffle=True,
        )
        f.create_dataset(
            "power",
            data=np.stack([s.power for s in samples]),
            chunks=(1, *meta.grid_shape),
            compression="gzip",
            shuffle=True,
        )
        f.create_dataset(
            "mask", data=np.stack(masks), chunks=(1, *meta.grid_shape), compression="gzip"
        )
        for name in ("xs", "rod_state"):
            if getattr(samples[0], name) is not None:
                f.create_dataset(
                    name,
                    data=np.stack([getattr(s, name) for s in samples]),
                    chunks=(1, *getattr(samples[0], name).shape),
                    compression="gzip",
                    shuffle=True,
                )
        if samples[0].solver_info is not None:
            f.create_dataset(
                "solver_info_json",
                data=[json.dumps(s.solver_info) for s in samples],
                dtype=h5py.string_dtype(),
            )
    (out_dir / "metadata.json").write_text(json.dumps(asdict(meta), indent=2))
    (out_dir / "samples.tmp.h5").replace(out_dir / "samples.h5")


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
