"""Stream compatible IID batches into one dataset, preserving the first split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import h5py

from datagen.iid_cases import configuration_hash, grouped_splits

PHYSICS_KEYS = (
    "solver",
    "solver_version",
    "n_groups",
    "grid_shape",
    "pitch_cm",
    "xs_fields",
    "bc",
    "boundary_gamma",
    "flux_normalization",
    "power_normalization",
    "physics",
    "assembly_pitch_cm",
    "material_names",
)


def shared_inputs(config):
    if "merged_shared_config" in config:
        return config["merged_shared_config"]
    if "sampling" not in config:
        raise ValueError("Merge only common-distribution IID batches, not the old stress families")
    return {
        "xs_table": config["xs_table"],
        "rodded_xs_table": config["rodded_xs_table"],
        "sampling_template_sha256": config["sampling_template_sha256"],
        "distribution": config["sampling"]["distribution"],
        "geometry": config["sampling"]["geometry"],
    }


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge(inputs: list[Path], out: Path, block_size: int = 16):
    inputs = [path.resolve() for path in inputs]
    if len(inputs) < 2 or len(set(inputs)) != len(inputs):
        raise ValueError("Provide at least two distinct dataset directories")
    if out.exists():
        raise FileExistsError("Merged output already exists; use a new output directory")
    metas = [json.loads((path / "metadata.json").read_text()) for path in inputs]
    configs = [json.loads((path / "inputs.json").read_text()) for path in inputs]
    shared = shared_inputs(configs[0])
    for meta, config in zip(metas[1:], configs[1:]):
        for key in PHYSICS_KEYS:
            if meta.get(key) != metas[0].get(key):
                raise ValueError(f"Incompatible datasets: {key}")
        if shared_inputs(config) != shared:
            raise ValueError("Incompatible material tables, geometry, or sampling distribution")
    with ExitStack() as stack:
        sources = [stack.enter_context(h5py.File(path / "samples.h5", "r")) for path in inputs]
        sizes = [len(source["k_eff"]) for source in sources]
        keys = set(sources[0])
        required = {"material_map", "rod_state", "xs", "mask", "flux", "power", "k_eff"}
        if not required.issubset(keys):
            raise ValueError("Dataset is missing required inputs or full-field outputs")
        for source, size in zip(sources, sizes):
            if set(source) != keys:
                raise ValueError("HDF5 fields differ")
            for key in keys:
                if (
                    source[key].shape != (size, *sources[0][key].shape[1:])
                    or source[key].dtype != sources[0][key].dtype
                ):
                    raise ValueError(f"Incompatible HDF5 shape/dtype: {key}")
        hashes = []
        for path, source, size in zip(inputs, sources, sizes):
            batch_hashes = []
            for start in range(0, size, block_size):
                ids = source["material_map"][start : start + block_size]
                rods = source["rod_state"][start : start + block_size]
                batch_hashes.extend(
                    configuration_hash({"material_map": a, "rod_state": b})
                    for a, b in zip(ids, rods)
                )
            saved = path / "input_hashes.json"
            if saved.exists() and json.loads(saved.read_text()) != batch_hashes:
                raise ValueError(f"Saved input hashes do not match actual data in {path}")
            hashes.extend(batch_hashes)
        # Preserve every original first-dataset assignment. Fresh extra-batch
        # split files are superseded; no old training row becomes a new test row.
        first_splits = json.loads((inputs[0] / "splits.json").read_text())
        if set(first_splits) != {"train", "val", "test"} or sorted(
            i for group in first_splits.values() for i in group
        ) != list(range(sizes[0])):
            raise ValueError("Invalid first-dataset split")
        assignments = {}
        for name, indices in first_splits.items():
            for i in indices:
                if hashes[i] in assignments and assignments[hashes[i]] != name:
                    raise ValueError("First dataset already has identical inputs across splits")
                assignments[hashes[i]] = name
        splits = {"train": [], "val": [], "test": []}
        for name, indices in grouped_splits(hashes, metas[0]["seed"]).items():
            for i in indices:
                splits[assignments.get(hashes[i], name)].append(i)
        total = sum(sizes)
        provenance = []
        seen_files = set()
        for path, size, config in zip(inputs, sizes, configs):
            sha = file_hash(path / "samples.h5")
            if sha in seen_files:
                raise ValueError("The same dataset content was supplied twice")
            seen_files.add(sha)
            provenance.append(
                {"path": str(path), "rows": size, "h5_sha256": sha, "original_inputs": config}
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp-", dir=out.parent))
        try:
            with h5py.File(temporary / "samples.h5", "w") as dest:
                for key in sorted(keys):
                    ref = sources[0][key]
                    dest.create_dataset(
                        key,
                        shape=(total, *ref.shape[1:]),
                        dtype=ref.dtype,
                        compression="gzip" if ref.ndim > 1 else None,
                    )
                offset = 0
                for source, size in zip(sources, sizes):
                    for start in range(0, size, block_size):
                        end = min(size, start + block_size)
                        for key in keys:
                            dest[key][offset + start : offset + end] = source[key][start:end]
                    offset += size
            meta = dict(metas[0], version=f"v2-iid-merged-{total}")
            report = {
                "rows": total,
                "unique_configurations": len(set(hashes)),
                "splits": {name: len(indices) for name, indices in splits.items()},
                "first_dataset_splits_preserved": True,
                "policy": "Retain IID repeats; all identical inputs share a split. Preserve the first dataset's assignments and assign new configurations with its split seed.",
                "sources": provenance,
            }
            documents = {
                "metadata.json": meta,
                "splits.json": splits,
                "input_hashes.json": hashes,
                "inputs.json": {
                    "generation_mode": "merged_independent_batches",
                    "merged_shared_config": shared,
                    "sources": provenance,
                },
                "merge_provenance.json": report,
            }
            for name, value in documents.items():
                (temporary / name).write_text(json.dumps(value, indent=2) + "\n")
            temporary.rename(out)
        except BaseException:
            shutil.rmtree(temporary)
            raise
    return {key: value for key, value in report.items() if key != "sources"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.inputs, args.out), indent=2))


if __name__ == "__main__":
    main()
