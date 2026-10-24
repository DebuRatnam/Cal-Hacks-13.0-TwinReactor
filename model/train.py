"""Train the surrogate.

Usage:
    python -m model.train --data data/v0 --out checkpoints/v0
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # TODO: load splits.json, CoreDataset for train/val, train loop, save checkpoint +
    # normalization stats + dataset version to args.out. Resumable from last checkpoint.
    raise NotImplementedError


if __name__ == "__main__":
    main()
