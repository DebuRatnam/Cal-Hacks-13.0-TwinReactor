"""Train-only scaling and reproducible subsets of the saved full-field dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from datagen.schema import read_metadata


def data_signature(data_dir):
    data_dir = Path(data_dir)
    meta = asdict(read_metadata(data_dir))
    payload = json.dumps(meta, sort_keys=True).encode()
    hashes = data_dir / "input_hashes.json"
    if hashes.exists():
        payload += hashes.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def choose_splits(data_dir, train_samples=500, seed=42):
    data_dir = Path(data_dir)
    splits = json.loads((data_dir / "splits.json").read_text())
    with h5py.File(data_dir / "samples.h5") as f:
        n = len(f["k_eff"])
    all_ids = [i for ids in splits.values() for i in ids]
    if sorted(all_ids) != list(range(n)):
        raise ValueError("Saved splits must be disjoint and cover the dataset")
    if not 1 <= train_samples <= len(splits["train"]):
        raise ValueError("train_samples exceeds the saved training split")
    if not splits["val"] or not splits["test"]:
        raise ValueError("Nonempty validation and test splits are required")
    if (data_dir / "input_hashes.json").exists():
        hashes = json.loads((data_dir / "input_hashes.json").read_text())
        seen = {}
        for split, ids in splits.items():
            for i in ids:
                if hashes[i] in seen and seen[hashes[i]] != split:
                    raise ValueError("An identical input appears in different saved splits")
                seen[hashes[i]] = split
    chosen = dict(splits)
    chosen["train"] = (
        np.random.default_rng(seed).permutation(splits["train"])[:train_samples].tolist()
    )
    return chosen


def load_cases(data_dir, indices):
    """Cache a small experiment in RAM; preserve arbitrary saved index ordering."""
    indices = list(map(int, indices))
    if not indices:
        raise ValueError("No cases selected")
    with h5py.File(Path(data_dir) / "samples.h5", "r") as f:
        if "xs" not in f or "mask" not in f:
            raise ValueError("Training requires resolved coefficient maps and a physical mask")
        arrays = {
            key: np.stack([f[key][i] for i in indices])
            for key in ("xs", "flux", "power", "mask", "k_eff")
        }
    x = arrays["xs"].astype(np.float32)
    fields = np.concatenate([arrays["flux"], arrays["power"][:, None]], axis=1).astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(fields).all():
        raise ValueError("Nonfinite training data")
    mask = arrays["mask"].astype(bool)
    fuel = mask & np.any(x[:, 4:6] > 0, axis=1)
    return {
        "x": x,
        "fields": fields,
        "k_eff": arrays["k_eff"].astype(np.float64),
        "mask": mask,
        "fuel_mask": fuel,
        "indices": np.array(indices),
    }


@dataclass
class Normalization:
    x_mean: list[float]
    x_std: list[float]
    field_mean: list[float]
    field_std: list[float]
    k_mean: float
    k_std: float

    @classmethod
    def fit(cls, train):
        # Each flux group has its own affine scale for conditioning; decode both
        # before physics checks so their original common physical ratio survives.
        x_values = train["x"].transpose(1, 0, 2, 3)[:, train["mask"]].astype(np.float64)
        means, stds = [], []
        for g in range(3):
            valid = train["mask"] if g < 2 else train["fuel_mask"]
            values = train["fields"][:, g][valid].astype(np.float64)
            means.append(0.0)
            stds.append(max(float(np.sqrt(np.mean(values**2))), 1e-12))
        delta = train["k_eff"] - 1
        return cls(
            x_values.mean(1).tolist(),
            np.maximum(x_values.std(1), 1e-12).tolist(),
            means,
            stds,
            float(delta.mean()),
            max(float(delta.std()), 1e-8),
        )

    def encode_x(self, x, mask):
        mean = np.asarray(self.x_mean, np.float32)[None, :, None, None]
        std = np.asarray(self.x_std, np.float32)[None, :, None, None]
        return np.where(mask[:, None], (x - mean) / std, 0).astype(np.float32)

    def encode_fields(self, fields, mask, fuel_mask):
        mean = np.asarray(self.field_mean, np.float32)[None, :, None, None]
        std = np.asarray(self.field_std, np.float32)[None, :, None, None]
        valid = np.stack([mask, mask, fuel_mask], axis=1)
        return np.where(valid, (fields - mean) / std, 0).astype(np.float32)

    def encode_k(self, k_eff):
        return ((k_eff - 1 - self.k_mean) / self.k_std).astype(np.float32)

    def decode(self, fields, k, mask, fuel_mask):
        mean = np.asarray(self.field_mean, np.float32)[None, :, None, None]
        std = np.asarray(self.field_std, np.float32)[None, :, None, None]
        valid = np.stack([mask, mask, fuel_mask], axis=1)
        physical = (fields * std + mean) * valid
        absolute_k = np.asarray(k).reshape(-1) * self.k_std + self.k_mean + 1
        return {"flux": physical[:, :2], "power": physical[:, 2], "k_eff": absolute_k}


def prepare(data_dir, train_samples=500, seed=42):
    meta = read_metadata(Path(data_dir))
    if meta.grid_orientation != "row0_south_col0_west":
        raise ValueError("Unsupported orientation")
    splits = choose_splits(data_dir, train_samples, seed)
    data = {name: load_cases(data_dir, ids) for name, ids in splits.items()}
    scaler = Normalization.fit(data["train"])
    return data, scaler, splits
