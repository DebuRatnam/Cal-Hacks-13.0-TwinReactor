"""Compare small-data baselines and neural operators on identical saved splits."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import torch

from model.baselines import MeanPredictor, PODBasis, fit_classical
from model.evaluate import latency_ms, physics_metrics, score_predictions, write_case_csv
from model.train import GridPredictor, train_model
from model.training_data import data_signature, prepare


def run(data_dir, out, train_samples=500, epochs=40, seed=42, threads=4):
    data_dir, out = Path(data_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    start = time.perf_counter()
    data, scaler, splits = prepare(data_dir, train_samples, seed)
    config = {
        "data_signature": data_signature(data_dir),
        "train_samples": train_samples,
        "seed": seed,
        "neural_epochs_max": epochs,
        "threads": threads,
    }
    if (out / "experiment.json").exists():
        previous = json.loads((out / "experiment.json").read_text())
        if previous["config"] != config:
            raise ValueError("Use a new output for changed experiment settings")
    report = {
        "config": config,
        "splits": splits,
        "normalization": asdict(scaler),
        "hardware": {"platform": platform.platform(), "device": "cpu", "torch_threads": threads},
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "numpy", "scikit-learn", "nvidia-physicsnemo")
        },
        "protocol": "One fixed seed. Same 500 training IDs, all saved validation/test IDs. Scaling/PCA fit on selected train only. Model/checkpoint selection uses validation; test only reports final metrics. Classical models use train-only input/output PCA; CNN/FNO predict full fields. Single-case latency includes scaling, prediction and field reconstruction, excludes disk I/O. No clipping or positivity correction of predictions.",
        "models": {},
    }
    (out / "experiment.json").write_text(json.dumps(report, indent=2))
    print(
        f"Loaded {len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test cases",
        flush=True,
    )
    print("Fitting train-only input/output PCA for classical baselines", flush=True)
    basis = PODBasis(scaler, seed=seed)
    x_train, y_train = basis.fit(data["train"])
    report["pca"] = {
        "input_modes": basis.x_pca.n_components_,
        "output_modes": basis.y_pca.n_components_,
        "input_explained_variance": float(basis.x_pca.explained_variance_ratio_.sum()),
        "output_explained_variance": float(basis.y_pca.explained_variance_ratio_.sum()),
        "fit_seconds": basis.fit_seconds,
        "test_oracle_reconstruction": score_predictions(
            basis.reconstruction(data["test"]), data["test"], scaler
        ),
        "oracle_note": "Uses true held-out output coefficients to measure compression error only; not a predictive model and not used for selection.",
    }
    for name in ("mean", "ridge", "extra_trees", "mlp", "cnn", "fno"):
        directory = out / name
        directory.mkdir(exist_ok=True)
        print(f"Starting {name}", flush=True)
        if name == "mean":
            clock = time.perf_counter()
            predictor = MeanPredictor(data["train"])
            fit = {"fit_seconds": time.perf_counter() - clock}
            joblib.dump(predictor, directory / "model.joblib")
        elif name in ("ridge", "extra_trees", "mlp"):
            path = directory / "model.joblib"
            if path.exists():
                saved = joblib.load(path)
                predictor = saved["predictor"]
                fit = {k: v for k, v in saved.items() if k not in ("predictor", "normalization")}
            else:
                predictor, fit = fit_classical(name, basis, data, x_train, y_train, path, seed=seed)
        else:
            if (directory / "training_result.json").exists():
                fit = json.loads((directory / "training_result.json").read_text())
            else:
                last = directory / "last.pt"
                fit = train_model(
                    data_dir,
                    directory,
                    model=name,
                    train_samples=train_samples,
                    epochs=epochs,
                    seed=seed,
                    prepared=(data, scaler, splits),
                    threads=threads,
                    resume=last if last.exists() else None,
                )
                (directory / "training_result.json").write_text(json.dumps(fit, indent=2))
            predictor = GridPredictor(directory / "best.pt")
        result = {"training": fit}
        for split in ("val", "test"):
            pred = predictor.predict(data[split])
            result[split] = score_predictions(pred, data[split], scaler)
            if split == "test":
                result[split].update(physics_metrics(pred, data[split], data_dir))
            write_case_csv(directory / f"{split}_predictions.csv", pred, data[split])
        result["latency_single_case_ms"] = latency_ms(predictor, data["test"])
        report["models"][name] = result
        (directory / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        (out / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(
            f"{name}: test k MAE={result['test']['k_mae_pcm']:.1f} pcm, fields={100 * result['test']['field_relative_l2_mean']:.2f}%, fit={fit['fit_seconds']:.1f}s",
            flush=True,
        )
    report["selected_by_validation"] = min(
        report["models"], key=lambda n: report["models"][n]["val"]["normalized_score"]
    )
    report["elapsed_seconds"] = time.perf_counter() - start
    (out / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model",
                "train_cases",
                "validation_score",
                "test_k_mae_pcm",
                "test_k_rmse_pcm",
                "test_fast_flux_rel_l2_pct",
                "test_thermal_flux_rel_l2_pct",
                "test_power_rel_l2_pct",
                "fit_seconds",
                "latency_single_case_ms",
            ]
        )
        for name, result in report["models"].items():
            t = result["test"]
            writer.writerow(
                [
                    name,
                    train_samples,
                    result["val"]["normalized_score"],
                    t["k_mae_pcm"],
                    t["k_rmse_pcm"],
                    100 * t["flux_fast_relative_l2_mean"],
                    100 * t["flux_thermal_relative_l2_mean"],
                    100 * t["power_relative_l2_mean"],
                    result["training"]["fit_seconds"],
                    result["latency_single_case_ms"],
                ]
            )
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-samples", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    report = run(args.data, args.out, args.train_samples, args.epochs, args.seed, args.threads)
    print("Selected by validation:", report["selected_by_validation"], flush=True)


if __name__ == "__main__":
    main()
