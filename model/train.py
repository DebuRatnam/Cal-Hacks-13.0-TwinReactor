"""Supervised full-field FNO training with train-only scaling and resume."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from datagen.schema import read_metadata
from model import build_model
from model.training_data import Normalization, data_signature, prepare


def compact_options(model):
    if model != "fno":
        raise ValueError(f"unknown model {model!r}; only 'fno' is available")
    return {"latent_channels": 16, "num_fno_layers": 3, "num_fno_modes": 8}


def tensor_data(cases, scaler):
    arrays = [
        scaler.encode_x(cases["x"], cases["mask"]),
        scaler.encode_fields(cases["fields"], cases["mask"], cases["fuel_mask"]),
        scaler.encode_k(cases["k_eff"]),
        cases["mask"],
        cases["fuel_mask"],
        cases["x"],
    ]
    return TensorDataset(*(torch.from_numpy(a) for a in arrays))


def supervised_loss(pred, fields, k, mask, fuel):
    values = torch.cat([pred["flux"], pred["power"][:, None]], dim=1)
    valid = torch.stack([mask, mask, fuel], dim=1)
    mse = ((values - fields).square() * valid).flatten(2).sum(2)
    mse = mse / valid.flatten(2).sum(2).clamp_min(1)
    return mse.mean(0).sum() + (pred["k"] - k).square().mean()


def train_model(
    data_dir,
    out,
    model="fno",
    train_samples=500,
    epochs=50,
    patience=10,
    seed=42,
    batch_size=16,
    lr=0.002,
    device="cpu",
    physics_weight=0.0,
    resume=None,
    model_options=None,
    prepared=None,
    threads=4,
):
    data_dir, out = Path(data_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and resume is None:
        raise FileExistsError("Checkpoint exists; use --resume or a new output directory")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    data, scaler, splits = (
        prepared if prepared is not None else prepare(data_dir, train_samples, seed)
    )
    options = model_options if model_options is not None else compact_options(model)
    config = {
        "model": model,
        "model_options": options,
        "in_channels": 7,
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "physics_weight": physics_weight,
        "train_samples": train_samples,
    }
    signature = data_signature(data_dir)
    meta = read_metadata(data_dir)
    net = build_model(model, in_channels=7, **options).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    gen = torch.Generator().manual_seed(seed)
    start_epoch, best, stale, history, elapsed = 0, float("inf"), 0, [], 0.0
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            saved["data_signature"] != signature
            or saved["config"] != config
            or saved["splits"] != splits
        ):
            raise ValueError("Resume data, subset or model configuration does not match")
        scaler = Normalization(**saved["normalization"])
        net.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        gen.set_state(saved["shuffle_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        start_epoch, best, stale = saved["epoch"], saved["best_val_loss"], saved["stale"]
        history, elapsed = saved["history"], saved["fit_seconds"]
    loaders = {
        name: DataLoader(
            tensor_data(data[name], scaler),
            batch_size=batch_size,
            shuffle=(name == "train"),
            generator=gen if name == "train" else None,
        )
        for name in ("train", "val")
    }
    (out / "normalization.json").write_text(json.dumps(asdict(scaler), indent=2))
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    field_mean = torch.tensor(scaler.field_mean, device=device)[None, :, None, None]
    field_std = torch.tensor(scaler.field_std, device=device)[None, :, None, None]
    start = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        scores = {}
        for split, loader in loaders.items():
            net.train(split == "train")
            total, count = 0.0, 0
            for tensors in loader:
                x, fields, k, mask, fuel, physical_xs = (v.to(device) for v in tensors)
                with torch.set_grad_enabled(split == "train"):
                    pred = net(x)
                    loss = supervised_loss(pred, fields, k, mask, fuel)
                    if physics_weight:
                        from model.physics import physics_loss

                        normalized = torch.cat([pred["flux"], pred["power"][:, None]], dim=1)
                        physical = normalized * field_std + field_mean
                        phi = physical[:, :2] * mask[:, None]
                        absolute_k = pred["k"] * scaler.k_std + scaler.k_mean + 1
                        loss = loss + physics_weight * physics_loss(
                            phi,
                            absolute_k.clamp_min(0.05),
                            physical_xs,
                            meta.pitch_cm,
                            bc=meta.bc,
                            gamma=meta.boundary_gamma,
                            void=~mask,
                        )
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite loss; check learning rate/scaling")
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        optimizer.step()
                total += float(loss.detach()) * len(x)
                count += len(x)
            scores[split] = total / count
        scheduler.step(scores["val"])
        improved = scores["val"] < best
        if improved:
            best, stale = scores["val"], 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": scores["train"],
                "val_loss": scores["val"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        checkpoint = {
            "format_version": 1,
            "config": config,
            "data_signature": signature,
            "metadata": asdict(meta),
            "normalization": asdict(scaler),
            "splits": splits,
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "shuffle_state": gen.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "epoch": epoch + 1,
            "best_val_loss": best,
            "stale": stale,
            "history": history,
            "fit_seconds": elapsed + time.perf_counter() - start,
        }
        for name in ["last.pt", "best.pt"] if improved else ["last.pt"]:
            tmp = out / (name + ".tmp")
            torch.save(checkpoint, tmp)
            tmp.replace(out / name)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"{model} epoch {epoch + 1}/{epochs}: train={scores['train']:.5f}, val={scores['val']:.5f}",
            flush=True,
        )
        if stale >= patience:
            break
    return {
        "checkpoint": str(out / "best.pt"),
        "fit_seconds": elapsed + time.perf_counter() - start,
        "best_val_loss": best,
        "epochs_completed": len(history),
        "parameters": sum(p.numel() for p in net.parameters()),
    }


class GridPredictor:
    def __init__(self, checkpoint, device="cpu"):
        self.saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.scaler = Normalization(**self.saved["normalization"])
        cfg = self.saved["config"]
        self.device = device
        self.net = build_model(
            cfg["model"], in_channels=cfg["in_channels"], **cfg["model_options"]
        ).to(device)
        self.net.load_state_dict(self.saved["state_dict"])
        self.net.eval()

    def predict(self, cases, batch_size=32):
        results = []
        for start in range(0, len(cases["x"]), batch_size):
            end = start + batch_size
            mask, fuel = cases["mask"][start:end], cases["fuel_mask"][start:end]
            x = self.scaler.encode_x(cases["x"][start:end], mask)
            with torch.no_grad():
                p = self.net(torch.from_numpy(x).to(self.device))
            fields = torch.cat([p["flux"], p["power"][:, None]], dim=1).cpu().numpy()
            results.append(self.scaler.decode(fields, p["k"].cpu().numpy(), mask, fuel))
        return {key: np.concatenate([r[key] for r in results]) for key in results[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", choices=["fno"], default="fno")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--train-samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--physics-weight", type=float, default=0)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()
    print(
        json.dumps(
            train_model(
                args.data,
                args.out,
                model=args.model,
                train_samples=args.train_samples,
                epochs=args.epochs,
                seed=args.seed,
                batch_size=args.batch_size,
                patience=args.patience,
                lr=args.lr,
                device=args.device,
                threads=args.threads,
                physics_weight=args.physics_weight,
                resume=args.resume,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
