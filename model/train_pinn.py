"""A separate, per-configuration PINN experiment; not a dataset operator."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from model.dataset import CoreDataset
from model.evaluate import score_predictions
from model.training_data import Normalization, load_cases


def run(data_dir, experiment_dir, out, steps=500, seed=42):
    data_dir, experiment_dir, out = Path(data_dir), Path(experiment_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    experiment = json.loads((experiment_dir / "experiment.json").read_text())
    # Chosen by saved sample index, without examining its reference solution.
    index = min(experiment["splits"]["val"])
    ds = CoreDataset(data_dir, [index])
    problem, _reference = ds.make_pinn_problem(0, interface_width_cm=0.5)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    model = problem.make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_gen = torch.Generator().manual_seed(seed)
    best = float("inf")
    best_state = None
    history = []
    start = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = problem.losses(model, n_interior=256, n_boundary=64, gen=train_gen)["total"]
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite PINN residual")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if (step + 1) % 50 == 0 or step == 0:
            # Fixed independent collocation points; selection never uses true flux/k.
            validation = problem.losses(
                model, n_interior=512, n_boundary=128, gen=torch.Generator().manual_seed(seed + 1)
            )
            score = float(validation["total"].detach())
            history.append({"step": step + 1, "collocation_loss": score})
            if score < best:
                best = score
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"PINN case {index}, step {step + 1}/{steps}: collocation={score:.5f}", flush=True
            )
    fit_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "sample_id": index,
            "steps": steps,
            "interface_width_cm": 0.5,
            "seed": seed,
            "history": history,
            "scope": "One configuration trained on PDE/BC/source-normalization residuals; not a learned layout operator",
        },
        out / "best.pt",
    )
    cases = load_cases(data_dir, [index])
    clock = time.perf_counter()
    phi = problem.evaluate_grid(model, n_per_cell=1) * cases["mask"][0][None]
    phi = phi / (phi.sum() * ds.meta.pitch_cm**2)
    with h5py.File(data_dir / "samples.h5") as f:
        kappa = f["kappa_fission"][index]
    power = (kappa * phi).sum(0)
    power = power / power[cases["fuel_mask"][0]].mean()
    pred = {
        "flux": phi[None],
        "power": power[None],
        "k_eff": np.array([float(model.k_eff.detach())]),
    }
    grid_seconds = time.perf_counter() - clock
    scaler = Normalization(**experiment["normalization"])
    report = {
        "sample_id": index,
        "steps": steps,
        "fit_seconds": fit_seconds,
        "grid_prediction_seconds": grid_seconds,
        "best_collocation_loss": best,
        "metrics_one_case": score_predictions(pred, cases, scaler),
        "caveat": "Single validation configuration, selected by index. Fixed PDE collocation points select checkpoint; reference labels only evaluate afterwards. Not comparable to a 500-case operator test. PINN smooths material interfaces and uses a fixed short optimization budget.",
    }
    (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    run(args.data, args.experiment, args.out, args.steps)


if __name__ == "__main__":
    main()
