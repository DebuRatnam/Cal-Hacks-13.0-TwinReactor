import numpy as np

from datagen.schema import DatasetMetadata, Sample, make_splits, read_metadata, write_dataset


def test_roundtrip(tmp_path):
    meta = DatasetMetadata(
        version="test", solver="ndiffusion", solver_version="0", xs_source="none",
        n_groups=2, grid_shape=(4, 4), pitch_cm=21.5, seed=0,
    )
    s = Sample(np.zeros((4, 4), int), 1.0, np.ones((2, 4, 4)), np.ones((4, 4)))
    write_dataset(tmp_path, meta, [s, s])
    assert read_metadata(tmp_path) == meta


def test_splits_disjoint_and_complete():
    sp = make_splits(100, seed=0)
    all_idx = sp["train"] + sp["val"] + sp["test"]
    assert sorted(all_idx) == list(range(100))
