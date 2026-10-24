"""Bulk training data generation with ndiffusion.

Usage:
    python -m datagen.ndiffusion_gen --config datagen/configs/quarter_core_small.yaml --out data/v0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from datagen.schema import Sample


def solve_case(material_map, params: dict) -> Sample:
    """Build nd.Materials + nd.KEigenSolver for one layout and return the solution.

    TODO: check ndiffusion docs (https://bwhewe-13.github.io/NeutronDiffusion/) for the 2-D XY
    solver signature and built-in power normalization / peaking factor helpers.
    """
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    # TODO: sample configs, solve (resumable, write shards as you go), write_dataset + splits.
    raise NotImplementedError(f"generation for {cfg['name']} not implemented")


if __name__ == "__main__":
    main()
