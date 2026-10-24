"""Core description the agent edits: assembly layout, material library, control-rod banks.

PLACEHOLDER PHYSICS (flag for EECS review before trusting any number):
- Cross sections are the IAEA 2-D benchmark materials with axial buckling Bz^2 = 0.8e-4 cm^-2 folded
  into removal, not the OpenMC-homogenized set the training data will use.
- A rod bank inserted a fraction f adds f * ROD_DELTA_SIG_R2 to thermal removal in its assemblies
  (IAEA "fuel 2 + rod" minus "fuel 2"). Fractional insertion of an axial rod is approximated by
  volume-weighting the rodded/unrodded XS of a 2-D slice; this ignores axial flux shape and
  overstates the worth of partially inserted banks.
- Power = nuSf1 phi1 + nuSf2 phi2 (constant kappa/nu assumed), assembly-averaged. Pin-level Fq and
  F_dH need pin-power reconstruction; only the assembly radial peaking factor is reported.

Grid convention matches model.physics: map[i][j], i = row = y (0 at the core-centre symmetry line,
"south"), j = column = x (0 at the west symmetry line). Reflective west/south, vacuum east/north.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

XS_FIELDS = ("D1", "D2", "sig_r1", "sig_r2", "nu_sig_f1", "nu_sig_f2", "sig_s12")
BZ2 = 0.8e-4  # cm^-2, IAEA 2-D axial buckling


def _iaea(D1, D2, sig_a1, sig_a2, nu_sig_f2, sig_s12) -> np.ndarray:
    return np.array(
        [D1, D2, sig_a1 + sig_s12 + D1 * BZ2, sig_a2 + D2 * BZ2, 0.0, nu_sig_f2, sig_s12], dtype=float
    )


MATERIALS: dict[str, dict] = {
    "fuel_low": {"xs": _iaea(1.5, 0.4, 0.01, 0.080, 0.135, 0.02), "fuel": True, "note": "IAEA fuel 1"},
    "fuel_high": {"xs": _iaea(1.5, 0.4, 0.01, 0.085, 0.135, 0.02), "fuel": True, "note": "IAEA fuel 2"},
    "reflector": {"xs": _iaea(2.0, 0.3, 0.0, 0.010, 0.0, 0.04), "fuel": False, "note": "IAEA reflector"},
}
ROD_DELTA_SIG_R2 = 0.045  # cm^-1


def _default_layout(n: int = 9) -> list[list[str]]:
    """Roughly cylindrical quarter core, symmetric about the diagonal."""
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            r = math.hypot(i + 0.5, j + 0.5)  # assembly pitches from core centre
            row.append("reflector" if r > 7.9 else "fuel_low" if r > 6.0 or (i + j) % 3 == 0 else "fuel_high")
        out.append(row)
    return out


DEFAULT_BANKS: dict[str, list[tuple[int, int]]] = {
    "A": [(0, 0)],
    "B": [(0, 4), (4, 0)],
    "C": [(2, 2), (4, 4)],
    "D": [(2, 6), (6, 2)],
}


@dataclass
class Core:
    layout: list[list[str]] = field(default_factory=_default_layout)
    banks: dict[str, list[tuple[int, int]]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_BANKS.items()})
    pitch_cm: float = 21.5

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.layout), len(self.layout[0])

    def fuel_mask(self, layout: list[list[str]] | None = None) -> np.ndarray:
        return np.array([[MATERIALS[m]["fuel"] for m in row] for row in (layout or self.layout)])

    def build_xs(
        self, rods: dict[str, float], changes: list[tuple[int, int, str]] = ()
    ) -> tuple[np.ndarray, list[list[str]]]:
        """Per-assembly XS [7, H, W] for a rod state {bank: inserted fraction 0..1} + material swaps."""
        layout = [list(row) for row in self.layout]
        H, W = self.shape
        for i, j, mat in changes:
            if mat not in MATERIALS:
                raise ValueError(f"unknown material {mat!r}; choose from {sorted(MATERIALS)}")
            if not (0 <= i < H and 0 <= j < W):
                raise ValueError(f"assembly ({i}, {j}) outside the {H}x{W} grid")
            layout[i][j] = mat
        xs = np.stack([np.stack([MATERIALS[m]["xs"] for m in row]) for row in layout]).transpose(2, 0, 1).copy()
        for bank, frac in rods.items():
            if bank not in self.banks:
                raise ValueError(f"unknown rod bank {bank!r}; banks are {sorted(self.banks)}")
            if not 0.0 <= frac <= 1.0:
                raise ValueError(f"bank {bank} insertion {frac} outside [0, 1]")
            for i, j in self.banks[bank]:
                if not MATERIALS[layout[i][j]]["fuel"]:
                    raise ValueError(f"bank {bank} position ({i}, {j}) is not a fuel assembly")
                xs[3, i, j] += frac * ROD_DELTA_SIG_R2
        return xs, layout
