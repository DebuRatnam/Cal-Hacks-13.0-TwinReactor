"""Create the comparison chart and human-readable report from a completed run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(root):
    root = Path(root)
    report = json.loads((root / "experiment.json").read_text())
    names = list(report["models"])
    labels = {
        "mean": "Mean",
        "ridge": "PCA + Ridge",
        "extra_trees": "PCA + Extra Trees",
        "mlp": "PCA + MLP",
        "cnn": "CNN",
        "fno": "FNO",
    }
    colors = [
        "#8C949C"
        if n == "mean"
        else "#D77B28"
        if n == "fno"
        else "#34876C"
        if n == "cnn"
        else "#2878A3"
        for n in names
    ]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.2), layout="constrained")
    measures = [
        ("Test k_eff MAE (pcm)", [report["models"][n]["test"]["k_mae_pcm"] for n in names]),
        (
            "Test mean field relative L2 (%)",
            [100 * report["models"][n]["test"]["field_relative_l2_mean"] for n in names],
        ),
        (
            "Single-case prediction (ms)",
            [report["models"][n]["latency_single_case_ms"] for n in names],
        ),
    ]
    for ax, (title, values) in zip(axes, measures):
        ax.barh(np.arange(len(names)), values, color=colors)
        ax.set(
            yticks=np.arange(len(names)),
            yticklabels=[labels[n] for n in names],
            xlabel=title,
            xlim=(0, max(values) * 1.22),
        )
        ax.invert_yaxis()
        for i, v in enumerate(values):
            ax.text(
                v + max(values) * 0.02,
                i,
                f"{v:,.2f}" if v < 10 else f"{v:,.1f}",
                va="center",
                fontsize=9,
            )
    fig.suptitle(
        f"{report['config']['train_samples']} training cases · {len(report['splits']['test'])} held-out test cases\nCPU benchmark, seed {report['config']['seed']} · lower is better",
        fontsize=14,
    )
    fig.savefig(root / "comparison.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    for ax, name in zip(axes, ["cnn", "fno"]):
        history = json.loads((root / name / "history.json").read_text())
        ax.plot(
            [h["epoch"] for h in history],
            [h["train_loss"] for h in history],
            label="Training",
            color="#2878A3",
        )
        ax.plot(
            [h["epoch"] for h in history],
            [h["val_loss"] for h in history],
            label="Validation",
            color="#D77B28",
        )
        ax.set(title=labels[name], xlabel="Epoch", ylabel="Scaled supervised loss", yscale="log")
        ax.legend(frameon=False)
    fig.savefig(root / "learning_curves.png", dpi=170)
    plt.close(fig)
    lines = [
        "# 500-case training benchmark",
        "",
        f"Selected by validation score: **{labels[report['selected_by_validation']]}**.",
        "",
        f"Training / validation / test: {len(report['splits']['train'])} / {len(report['splits']['val'])} / {len(report['splits']['test'])}. Seed {report['config']['seed']}. CPU, {report['hardware']['torch_threads']} Torch threads.",
        "",
        "| Model | k MAE (pcm) | Fast flux L2 (%) | Thermal flux L2 (%) | Power L2 (%) | Fit (s) | Prediction (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n, r in report["models"].items():
        t = r["test"]
        lines.append(
            f"| {labels[n]} | {t['k_mae_pcm']:.1f} | {100 * t['flux_fast_relative_l2_mean']:.2f} | {100 * t['flux_thermal_relative_l2_mean']:.2f} | {100 * t['power_relative_l2_mean']:.2f} | {r['training']['fit_seconds']:.1f} | {r['latency_single_case_ms']:.2f} |"
        )
    oracle = report["pca"]["test_oracle_reconstruction"]
    lines.extend(
        [
            "",
            "Classical pipelines use train-only input PCA (64 modes), output PCA (40 modes), and separate field/k regressors. CNN/FNO use the full coefficient grids and predict full fields. This compares practical pipelines, not PCA's isolated effect on a fixed architecture.",
            "",
            f"Output PCA training explained variance: {100 * report['pca']['output_explained_variance']:.3f}%. Held-out oracle field reconstruction error: {100 * oracle['field_relative_l2_mean']:.3f}% mean relative L2. The oracle uses true output coefficients and is not a predictive baseline.",
            "",
            "All scaling/PCA fits use only the selected training cases. Saved validation selects hyperparameters/checkpoints; the test split only reports final performance. No negative-value clipping or post-hoc field renormalization is used for operator predictions. Latency includes scaling and PCA reconstruction where applicable, excludes disk I/O, and is a warmed median over 20 single-case calls. Classical fit times include shared PCA preprocessing; neural fit times include validation and checkpoint writes.",
            "",
            "This is one small-data run with compact architectures and fixed budgets, not a fully tuned or multi-seed ranking. Results do not establish safety-grade accuracy or full-data scaling. PINN is a separate per-configuration solver; see pinn/result.json if that experiment was run. The existing runtime agent's IAEA placeholder still needs alignment before attaching these checkpoints.",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    render(parser.parse_args().experiment)
