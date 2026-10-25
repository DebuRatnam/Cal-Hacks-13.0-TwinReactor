"""Physical-unit metrics for predictions and standalone checkpoint evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from datagen.schema import read_metadata
from model.train import GridPredictor
from model.training_data import Normalization, data_signature, load_cases


def score_predictions(pred, cases, scaler):
    true = cases["fields"]
    fields = np.concatenate([pred["flux"], pred["power"][:, None]], axis=1)
    if not np.isfinite(fields).all() or not np.isfinite(pred["k_eff"]).all():
        raise ValueError("Nonfinite model prediction")
    valid = np.stack([cases["mask"], cases["mask"], cases["fuel_mask"]], axis=1)
    difference = (fields - true) * valid
    relative = np.sqrt(
        np.sum(difference.astype(np.float64) ** 2, axis=(2, 3))
        / np.maximum(np.sum((true * valid).astype(np.float64) ** 2, axis=(2, 3)), 1e-30)
    )
    k_error = (pred["k_eff"] - cases["k_eff"]) * 1e5
    normalized = difference / np.asarray(scaler.field_std)[None, :, None, None]
    field_mse = np.sum(normalized**2, axis=(2, 3)) / np.maximum(valid.sum(axis=(2, 3)), 1)
    k_mse = ((pred["k_eff"] - cases["k_eff"]) / scaler.k_std) ** 2
    groups = {}
    for name, selection in {
        "subcritical_below_0.99": cases["k_eff"] < 0.99,
        "near_critical_0.99_to_1.01": np.abs(cases["k_eff"] - 1) <= 0.01,
        "supercritical_above_1.01": cases["k_eff"] > 1.01,
    }.items():
        groups[name] = {
            "count": int(selection.sum()),
            "k_mae_pcm": float(np.abs(k_error[selection]).mean()) if selection.any() else None,
        }
    return {
        "samples": len(k_error),
        "k_mae_pcm": float(np.abs(k_error).mean()),
        "k_rmse_pcm": float(np.sqrt(np.mean(k_error**2))),
        "k_p95_abs_pcm": float(np.quantile(np.abs(k_error), 0.95)),
        "flux_fast_relative_l2_mean": float(relative[:, 0].mean()),
        "flux_thermal_relative_l2_mean": float(relative[:, 1].mean()),
        "power_relative_l2_mean": float(relative[:, 2].mean()),
        "field_relative_l2_mean": float(relative.mean()),
        "field_relative_l2_p95": float(np.quantile(relative.mean(1), 0.95)),
        "negative_flux_fraction": float((pred["flux"][:, :, cases["mask"][0]] < 0).mean())
        if np.all(cases["mask"] == cases["mask"][0])
        else float(((fields[:, :2] < 0) * valid[:, :2]).sum() / valid[:, :2].sum()),
        "normalized_score": float(field_mse.sum(1).mean() + k_mse.mean()),
        "criticality_groups": groups,
    }


def physics_metrics(pred, cases, data_dir):
    from model.physics import physics_loss, rayleigh_k

    meta = read_metadata(Path(data_dir))
    losses, k_gaps, integral_errors = [], [], []
    for start in range(0, len(cases["x"]), 64):
        end = start + 64
        phi = torch.from_numpy(pred["flux"][start:end]).double()
        xs = torch.from_numpy(cases["x"][start:end]).double()
        k = torch.from_numpy(np.asarray(pred["k_eff"][start:end])).double()
        mask = torch.from_numpy(cases["mask"][start:end])
        kw = {"h_cm": meta.pitch_cm, "bc": meta.bc, "gamma": meta.boundary_gamma, "void": ~mask}
        losses.append(float(physics_loss(phi, k, xs, **kw)))
        kr = rayleigh_k(phi, xs, **kw).numpy()
        k_gaps.extend((np.abs(kr - np.asarray(pred["k_eff"][start:end])) * 1e5).tolist())
        integral_errors.extend(
            np.abs(phi.numpy().sum(axis=(1, 2, 3)) * meta.pitch_cm**2 - 1).tolist()
        )
    finite = np.isfinite(k_gaps)
    return {
        "fv_relative_squared_residual_batch_mean": float(np.mean(losses)),
        "rayleigh_vs_head_mae_pcm": float(np.mean(np.asarray(k_gaps)[finite]))
        if finite.any()
        else None,
        "rayleigh_nonfinite_fraction": float(1 - finite.mean()),
        "flux_integral_mae": float(np.mean(integral_errors)),
    }


def latency_ms(predictor, cases, repeats=20):
    one = {key: value[:1] for key, value in cases.items()}
    for _ in range(3):
        predictor.predict(one)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        predictor.predict(one)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def write_case_csv(path, pred, cases):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "k_true", "k_pred", "error_pcm"])
        writer.writerows(
            (int(i), float(t), float(p), float((p - t) * 1e5))
            for i, t, p in zip(cases["indices"], cases["k_eff"], pred["k_eff"])
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--out", type=Path, default=Path("evaluation.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ckpt = args.ckpt / "best.pt" if args.ckpt.is_dir() else args.ckpt
    predictor = GridPredictor(ckpt, args.device)
    if predictor.saved["data_signature"] != data_signature(args.data):
        raise ValueError("Checkpoint and evaluation data contract do not match")
    cases = load_cases(args.data, predictor.saved["splits"][args.split])
    pred = predictor.predict(cases)
    scaler = Normalization(**predictor.saved["normalization"])
    report = score_predictions(pred, cases, scaler)
    report.update(physics_metrics(pred, cases, args.data))
    report["latency_single_case_ms"] = latency_ms(predictor, cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    write_case_csv(args.out.with_suffix(".csv"), pred, cases)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
