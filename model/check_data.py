"""Audit stored full fields and exercise the real architecture interfaces without training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from datagen.physics_contract import validate_xs
from model.dataset import CoreDataset


def check(data_dir: Path, models: bool = False) -> dict:
    ds = CoreDataset.from_split(data_dir, "train")
    splits = json.loads((data_dir / "splits.json").read_text())
    with h5py.File(data_dir / "samples.h5", "r") as f:
        n = len(f["k_eff"])
        assert sorted(i for indices in splits.values() for i in indices) == list(range(n)), (
            "Overlapping/incomplete splits"
        )
        max_integral_error, max_power_error = 0.0, 0.0
        # Batched reads also work efficiently on the original multi-row chunks.
        for start in range(0, n, 128):
            blocks = {
                key: f[key][start : start + 128] for key in ("xs", "mask", "flux", "power", "k_eff")
            }
            for xs, mask, phi, power, k in zip(
                *(blocks[key] for key in ("xs", "mask", "flux", "power", "k_eff"))
            ):
                validate_xs(xs, mask)
                assert np.isfinite(phi).all() and np.isfinite(power).all()
                assert (phi >= 0).all() and (power >= 0).all()
                assert np.isfinite(k) and k > 0
                assert (phi[:, ~mask] == 0).all() and (power[~mask] == 0).all()
                fuel = mask & np.any(xs[4:6] > 0, axis=0)
                assert (power[~fuel] == 0).all()
                max_integral_error = max(
                    max_integral_error, abs(float(phi.sum() * ds.meta.pitch_cm**2) - 1)
                )
                max_power_error = max(max_power_error, abs(float(power[fuel].mean()) - 1))
            if "solver_info_json" in f:
                infos = f["solver_info_json"].asstr()[start : start + 128]
                assert all(json.loads(info)["converged"] for info in infos)
        assert max_integral_error < 1e-5 and max_power_error < 1e-5
        report = {
            "samples": n,
            "input_shape": [n, ds.in_channels, *ds.meta.grid_shape],
            "flux_shape": list(f["flux"].shape),
            "splits": {key: len(value) for key, value in splits.items()},
            "max_flux_integral_error": max_integral_error,
            "max_fuel_mean_power_error": max_power_error,
            "scope": ds.meta.scope,
            "grid_orientation": ds.meta.grid_orientation,
            "pitch_cm": ds.meta.pitch_cm,
            "boundary_gamma": ds.meta.boundary_gamma,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                key: importlib.metadata.version(key)
                for key in ("ndiffusion", "numpy", "h5py", "torch")
            },
        }
    if not models:
        return report

    from model import build_model
    from model.physics import fv_residual, physics_loss, rayleigh_k

    torch.manual_seed(0)
    batch = next(iter(DataLoader(ds, batch_size=min(2, len(ds)))))
    report["architectures"] = {}
    for name, opts in (
        ("fno", {"latent_channels": 8, "num_fno_layers": 1, "num_fno_modes": 3}),
        ("cnn", {"conv_layer_size": 8, "n_blocks": 1, "n_downsampling": 1}),
    ):
        net = build_model(name, in_channels=ds.in_channels, **opts)
        pred = net(batch["x"])
        for key in ("flux", "power", "k"):
            assert pred[key].shape == batch[key].shape
        loss = sum(value.square().mean() for value in pred.values())
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)
        report["architectures"][name] = {
            "forward_and_backward": "passed",
            "output_shapes": {key: list(value.shape) for key, value in pred.items()},
        }

    # Check every solver field against the partner's finite-volume equation,
    # using native float64 labels rather than the float32 training view.
    residuals = []
    max_sample_residual, max_rayleigh_error = 0.0, 0.0
    with h5py.File(data_dir / "samples.h5", "r") as f:
        for start in range(0, n, 64):
            b = {
                key: torch.from_numpy(f[key][start : start + 64])
                for key in ("flux", "k_eff", "xs", "mask")
            }
            kw = ds.physics_kwargs(b)
            residuals.append(float(physics_loss(b["flux"], b["k_eff"], b["xs"], **kw)))
            r = fv_residual(b["flux"], b["k_eff"], b["xs"], **kw)
            source = b["xs"][:, 4] * b["flux"][:, 0] + b["xs"][:, 5] * b["flux"][:, 1]
            per_sample = r.square().flatten(1).sum(1) / source.square().flatten(1).sum(1).clamp_min(
                1e-30
            )
            max_sample_residual = max(max_sample_residual, float(per_sample.max()))
            kr = rayleigh_k(b["flux"], b["xs"], **kw)
            max_rayleigh_error = max(max_rayleigh_error, float((kr - b["k_eff"]).abs().max()))
    report["max_batch_mean_relative_squared_fv_residual"] = max(residuals)
    report["max_sample_relative_squared_fv_residual"] = max_sample_residual
    report["max_absolute_rayleigh_k_error"] = max_rayleigh_error
    assert max_sample_residual < 1e-10
    assert max_rayleigh_error < 1e-6
    interface_width_cm = 0.5
    problem, reference = ds.make_pinn_problem(0, interface_width_cm=interface_width_cm)
    pinn = problem.make_model(layer_size=16, num_layers=2)
    losses = problem.losses(pinn, n_interior=32, n_boundary=16)
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert pinn.log_k.grad is not None and torch.isfinite(pinn.log_k.grad)
    report["architectures"]["pinn"] = {
        "forward_and_backward": "passed",
        "mode": "one configuration",
        "reference_points": list(reference["flux_points"].shape),
        "interface_width_cm": interface_width_cm,
        "caveat": "Existing PINN smooths material interfaces; this check verifies its data interface, not trained accuracy",
    }
    report["packages"]["nvidia-physicsnemo"] = importlib.metadata.version("nvidia-physicsnemo")
    report["training_performed"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = check(args.data, args.models)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
