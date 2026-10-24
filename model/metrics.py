"""Evaluation metrics in reactor-physics units."""

from __future__ import annotations

import numpy as np


def k_error_pcm(k_pred: np.ndarray, k_true: np.ndarray) -> np.ndarray:
    """Reactivity-free k error in pcm (1 pcm = 1e-5)."""
    return (k_pred - k_true) * 1e5


def power_rel_error(p_pred: np.ndarray, p_true: np.ndarray, fuel_mask: np.ndarray) -> dict:
    """Max and RMS relative power error over fuel cells."""
    rel = (p_pred[..., fuel_mask] - p_true[..., fuel_mask]) / p_true[..., fuel_mask]
    return {"max": float(np.abs(rel).max()), "rms": float(np.sqrt(np.mean(rel**2)))}


def peaking_factor(power: np.ndarray, fuel_mask: np.ndarray) -> float:
    """Max / mean power over fuel cells."""
    p = power[fuel_mask]
    return float(p.max() / p.mean())
