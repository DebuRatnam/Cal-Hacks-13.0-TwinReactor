"""PyTorch Dataset over a generated dataset directory."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class CoreDataset(Dataset):
    def __init__(self, data_dir: Path, indices: list[int], n_materials: int):
        with h5py.File(data_dir / "samples.h5", "r") as f:
            self.material_map = f["material_map"][indices]
            self.k_eff = f["k_eff"][indices]
            self.flux = f["flux"][indices]
            self.power = f["power"][indices]
        self.n_materials = n_materials
        # TODO: compute/load normalization stats from the train split only.

    def __len__(self) -> int:
        return len(self.k_eff)

    def __getitem__(self, i: int):
        x = np.eye(self.n_materials, dtype=np.float32)[self.material_map[i]].transpose(2, 0, 1)
        return {
            "x": torch.from_numpy(x),
            "k": torch.tensor(self.k_eff[i] - 1.0, dtype=torch.float32),
            "flux": torch.from_numpy(self.flux[i].astype(np.float32)),
            "power": torch.from_numpy(self.power[i].astype(np.float32)),
        }
