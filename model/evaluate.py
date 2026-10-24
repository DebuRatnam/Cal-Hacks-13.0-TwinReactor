"""Evaluate a checkpoint on the test split: k error (pcm), power error, peaking factor error,
plus a trivial baseline for comparison.

Usage:
    python -m model.evaluate --data data/v0 --ckpt checkpoints/v0
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    args = ap.parse_args()
    # TODO: run model + mean-prediction baseline, report model.metrics, physics checks
    # (flux >= 0, symmetry, zero power in reflector).
    raise NotImplementedError


if __name__ == "__main__":
    main()
