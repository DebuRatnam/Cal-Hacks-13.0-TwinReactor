"""Leakage, mask, physical scaling, checkpoint and resume regression checks."""

import json

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from datagen.schema import DatasetMetadata, write_dataset
from model.training_data import Normalization, choose_splits, load_cases, prepare


@pytest.fixture
def tiny_data(tmp_path):
    from datagen.ndiffusion_gen import solve_case

    samples = []
    for i in range(12):
        table = np.array([[1.5, 0.4, 0.03, 0.07 + i * 0.003, 0, 0.12 + i * 0.004, 0.02]])
        samples.append(
            solve_case(
                np.zeros((6, 8), int),
                {
                    "xs_table": table,
                    "pitch_cm": 2.5,
                    "bc": {
                        "west": "reflective",
                        "east": "reflective",
                        "south": "reflective",
                        "north": "reflective",
                    },
                },
            )
        )
    meta = DatasetMetadata(
        "training-test",
        "ndiffusion",
        "1.1.0",
        "synthetic",
        2,
        (6, 8),
        2.5,
        42,
        flux_normalization="unit_integral_total_2group",
        bc={
            "west": "reflective",
            "east": "reflective",
            "south": "reflective",
            "north": "reflective",
        },
    )
    write_dataset(tmp_path, meta, samples)
    (tmp_path / "splits.json").write_text(
        json.dumps({"train": list(range(8)), "val": [8, 9], "test": [10, 11]})
    )
    return tmp_path


def test_scaling_roundtrip_and_train_only_fit(tiny_data):
    data, scaler, splits = prepare(tiny_data, 6, 42)
    assert len(splits["train"]) == 6
    assert set(splits["train"]) <= set(range(8))
    assert splits == choose_splits(tiny_data, 6, 42)
    x = data["train"]
    fields = scaler.encode_fields(x["fields"], x["mask"], x["fuel_mask"])
    decoded = scaler.decode(fields, scaler.encode_k(x["k_eff"]), x["mask"], x["fuel_mask"])
    np.testing.assert_allclose(decoded["flux"], x["fields"][:, :2], rtol=1e-6)
    np.testing.assert_allclose(decoded["k_eff"], x["k_eff"], rtol=1e-6)
    before = scaler.x_mean.copy()
    data["val"]["x"] *= 1000
    assert Normalization.fit(data["train"]).x_mean == before
    assert not np.shares_memory(scaler.encode_x(x["x"], x["mask"]), x["x"])


def test_supervised_loss_excludes_void_and_nonfuel():
    from model.train import supervised_loss

    pred = {"flux": torch.zeros(1, 2, 2, 2), "power": torch.zeros(1, 2, 2), "k": torch.zeros(1)}
    mask = torch.tensor([[[True, True], [True, False]]])
    fuel = torch.tensor([[[True, False], [True, False]]])
    pred["flux"][:, :, 1, 1] = 999
    pred["power"][:, 0, 1] = 999
    pred["power"][:, 1, 1] = 999
    assert float(supervised_loss(pred, torch.zeros(1, 3, 2, 2), torch.zeros(1), mask, fuel)) == 0


def test_fno_checkpoint_resume_and_evaluation(tiny_data, tmp_path):
    pytest.importorskip("physicsnemo")
    from model.evaluate import score_predictions
    from model.train import GridPredictor, train_model

    out = tmp_path / "checkpoint"
    options = {"latent_channels": 4, "num_fno_layers": 1, "num_fno_modes": 3}
    first = train_model(
        tiny_data, out, train_samples=6, epochs=1, batch_size=2, model_options=options, threads=1
    )
    predictor = GridPredictor(out / "best.pt")
    cases = load_cases(tiny_data, [11, 10])
    pred = predictor.predict(cases)
    assert pred["flux"].shape == (2, 2, 6, 8)
    report = score_predictions(pred, cases, predictor.scaler)
    assert np.isfinite(report["normalized_score"])
    assert first["epochs_completed"] == 1
    second = train_model(
        tiny_data,
        out,
        train_samples=6,
        epochs=2,
        batch_size=2,
        model_options=options,
        threads=1,
        resume=out / "last.pt",
    )
    assert second["epochs_completed"] == 2
    with pytest.raises(ValueError, match="does not match"):
        train_model(
            tiny_data,
            out,
            train_samples=5,
            epochs=3,
            batch_size=2,
            model_options=options,
            threads=1,
            resume=out / "last.pt",
        )
    with h5py.File(tiny_data / "samples.h5") as f:
        np.testing.assert_array_equal(cases["x"][0], f["xs"][11].astype(np.float32))
