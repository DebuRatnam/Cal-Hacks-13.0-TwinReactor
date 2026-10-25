"""Full-field data for the FNO surrogate."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from datagen.physics_contract import N_XS
from datagen.schema import read_metadata


class CoreDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        indices: list[int],
        n_materials: int | None = None,
        features: str = "auto",
        include_mask: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.meta = read_metadata(self.data_dir)
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        with h5py.File(self.data_dir / "samples.h5", "r") as f:
            n = len(f["k_eff"])
            if ((self.indices < 0) | (self.indices >= n)).any():
                raise IndexError("Sample index outside dataset")
            has_xs = "xs" in f
            if n_materials is None:
                n_materials = len(self.meta.material_names)
                if not n_materials and (features == "onehot" or not has_xs):
                    n_materials = int(f["material_map"][:].max()) + 1
        if self.meta.grid_orientation != "row0_south_col0_west":
            raise ValueError("Unsupported grid orientation; use row 0 south, column 0 west")
        self.features = ("xs" if has_xs else "onehot") if features == "auto" else features
        if self.features not in ("xs", "onehot") or (self.features == "xs" and not has_xs):
            raise ValueError("XS features require a dataset containing resolved cross sections")
        if self.features == "onehot" and not n_materials:
            raise ValueError("One-hot features require a positive material count")
        self.n_materials = n_materials
        self.include_mask = include_mask
        self.in_channels = (N_XS if self.features == "xs" else self.n_materials) + int(include_mask)

    @classmethod
    def from_split(cls, data_dir: Path, split: str, **kwargs):
        splits = json.loads((Path(data_dir) / "splits.json").read_text())
        return cls(data_dir, splits[split], **kwargs)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        # Read one sample: saved random split ordering and separate workers do
        # not hit h5py's sorted fancy-index restriction or share an open file.
        idx = int(self.indices[i])
        with h5py.File(self.data_dir / "samples.h5", "r") as f:
            ids = f["material_map"][idx]
            mask = f["mask"][idx].astype(bool) if "mask" in f else ids >= 0
            flux, power = f["flux"][idx], f["power"][idx]
            k_eff = float(f["k_eff"][idx])
            xs = f["xs"][idx].astype(np.float32) if "xs" in f else None
            rod = f["rod_state"][idx] if "rod_state" in f else np.zeros(mask.shape)
        if self.features == "xs":
            x = xs.copy()
        else:
            if (ids[mask] < 0).any() or (ids[mask] >= self.n_materials).any():
                raise ValueError("Active material ID outside one-hot vocabulary")
            x = np.zeros((self.n_materials, *ids.shape), np.float32)
            x[:, mask] = np.eye(self.n_materials, dtype=np.float32)[ids[mask]].T
        if self.include_mask:
            x = np.concatenate([x, mask[None].astype(np.float32)])
        out = {
            "x": torch.from_numpy(x),
            "k": torch.tensor(k_eff - 1.0, dtype=torch.float32),
            "flux": torch.from_numpy(flux.astype(np.float32)),
            "power": torch.from_numpy(power.astype(np.float32)),
            "mask": torch.from_numpy(mask),
            "material_map": torch.from_numpy(ids.astype(np.int64)),
            "rod_state": torch.from_numpy(np.asarray(rod, dtype=np.float32)),
            "sample_id": torch.tensor(idx, dtype=torch.int64),
        }
        if xs is not None:
            out["xs"] = torch.from_numpy(xs)
            out["fuel_mask"] = torch.from_numpy(mask & np.any(xs[4:6] > 0, axis=0))
        return out

    def physics_kwargs(self, batch):
        """Use with physics_loss rather than guessing boundary defaults."""
        return {
            "h_cm": self.meta.pitch_cm,
            "bc": self.meta.bc,
            "gamma": self.meta.boundary_gamma,
            "void": ~batch["mask"],
        }
