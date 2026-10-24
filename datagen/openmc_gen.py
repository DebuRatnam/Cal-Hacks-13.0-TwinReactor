"""OpenMC runs: 2-group assembly homogenization and high-fidelity validation cases.

Requires OpenMC (conda) and OPENMC_CROSS_SECTIONS set.
2-D models use reflective boundaries in z. Always record k_eff std dev and check Shannon entropy.
"""

from __future__ import annotations


def homogenize_assembly(enrichment_wt: float, boron_ppm: float) -> dict:
    """Run an infinite-lattice assembly and return 2-group XS (D, removal, scatter, nusigf, chi)."""
    raise NotImplementedError


def run_core(material_map, params: dict) -> dict:
    """Full 2-D core run for the validation set. Returns k_eff, k_std, flux, power."""
    raise NotImplementedError
