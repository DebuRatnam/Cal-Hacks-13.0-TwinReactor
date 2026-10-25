"""Storage/split tests using small synthetic fixtures; no physics solves."""

import json

import h5py
import numpy as np
import pytest

from datagen.iid_cases import configuration_hash
from datagen.merge_datasets import merge
from datagen.schema import DatasetMetadata, Sample, write_dataset


def batch(path, codes, seed, splits):
    samples = []
    hashes = []
    for code in codes:
        ids = np.ones((2, 2), dtype=int)
        ids[0, 0] = code % 2
        rod = np.zeros((2, 2))
        rod[1, 1] = code // 2
        xs = np.broadcast_to(
            np.array([1.0, 0.4, 0.1, 0.1, 0.2, 0.1, 0.02])[:, None, None], (7, 2, 2)
        ).copy()
        samples.append(
            Sample(
                ids,
                1 + 0.01 * code,
                np.full((2, 2, 2), 0.125),
                np.ones((2, 2)),
                xs,
                rod,
                np.ones((2, 2), bool),
                {"fixture": True},
            )
        )
        hashes.append(configuration_hash({"material_map": ids, "rod_state": rod}))
    meta = DatasetMetadata("fixture", "ndiffusion", "1.1.0", "fixture", 2, (2, 2), 1.0, seed)
    write_dataset(path, meta, samples)
    config = {
        "sampling": {"distribution": "uniform_count_then_uniform_subset", "geometry": "quarter"},
        "seed": seed,
        "xs_table": [[1]],
        "rodded_xs_table": [[2]],
        "sampling_template_sha256": "fixture-template",
    }
    (path / "inputs.json").write_text(json.dumps(config))
    (path / "splits.json").write_text(json.dumps(splits))
    (path / "input_hashes.json").write_text(json.dumps(hashes))
    return path


@pytest.fixture
def sources(tmp_path):
    a = batch(tmp_path / "a", [0, 1], 10, {"train": [0], "val": [], "test": [1]})
    b = batch(tmp_path / "b", [1, 2], 11, {"train": [0, 1], "val": [], "test": []})
    return a, b


def test_merge_copies_full_fields_and_preserves_old_holdout(sources, tmp_path):
    a, b = sources
    before = (a / "samples.h5").read_bytes()
    out = tmp_path / "merged"
    report = merge([a, b], out, block_size=1)
    assert report["rows"] == 4 and report["unique_configurations"] == 3
    splits = json.loads((out / "splits.json").read_text())
    assert 0 in splits["train"]
    assert 1 in splits["test"] and 2 in splits["test"]
    assert 2 not in splits["train"]
    with h5py.File(out / "samples.h5") as merged:
        for key in merged:
            with h5py.File(a / "samples.h5") as first, h5py.File(b / "samples.h5") as second:
                expected = np.concatenate([first[key][:], second[key][:]])
                np.testing.assert_array_equal(merged[key][:], expected)
    assert (a / "samples.h5").read_bytes() == before


@pytest.mark.parametrize(
    "key,value",
    [("pitch_cm", 2.0), ("flux_normalization", "different"), ("boundary_gamma", 0.4692)],
)
def test_incompatible_physics_rejected(sources, tmp_path, key, value):
    a, b = sources
    meta = json.loads((b / "metadata.json").read_text())
    meta[key] = value
    (b / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Incompatible"):
        merge([a, b], tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_different_material_catalog_rejected(sources, tmp_path):
    a, b = sources
    cfg = json.loads((b / "inputs.json").read_text())
    cfg["xs_table"] = [[9]]
    (b / "inputs.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="material tables"):
        merge([a, b], tmp_path / "merged")


def test_repeated_paths_and_existing_output_rejected(sources, tmp_path):
    a, b = sources
    with pytest.raises(ValueError, match="distinct"):
        merge([a, a], tmp_path / "merged")
    merge([a, b], tmp_path / "merged")
    with pytest.raises(FileExistsError):
        merge([a, b], tmp_path / "merged")


def test_saved_input_hashes_checked(sources, tmp_path):
    a, b = sources
    (b / "input_hashes.json").write_text(json.dumps(["bad", "bad"]))
    with pytest.raises(ValueError, match="hashes"):
        merge([a, b], tmp_path / "merged")


def test_already_merged_dataset_can_be_extended(sources, tmp_path):
    a, b = sources
    merged = tmp_path / "merged"
    merge([a, b], merged)
    c = batch(tmp_path / "c", [3], 12, {"train": [0], "val": [], "test": []})
    report = merge([merged, c], tmp_path / "extended")
    assert report["rows"] == 5
    assert report["unique_configurations"] == 4
