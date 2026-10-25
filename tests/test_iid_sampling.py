"""Input-only checks. None of these tests run a neutron-diffusion solve."""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from datagen.iid_cases import IIDCases, configuration_hash, grouped_splits, main

CONFIG = Path(__file__).parents[1] / "datagen/configs/iaea_iid.yaml"


def sampler(n=1000, geometry="quarter"):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["n_samples"] = n
    cfg["sampling"]["geometry"] = geometry
    return IIDCases(cfg, CONFIG.parent)


def test_rod_sites_match_reference_and_material_endpoints():
    cases = sampler()
    assert len(cases.rod_sites) == 4
    assert cases[0]["material_map"].shape == (68, 68)
    for i in range(100):
        ids, rod = cases.block_case(i)
        assert np.isin(rod, [0, 1]).all()
        assert np.all(rod[~cases.eligible] == 0)
        assert np.all(ids[cases.eligible] == 1)
        fixed = ~np.isin(cases.base, [0, 1])
        assert np.array_equal(ids[fixed], cases.base[fixed])
        assert np.isin(ids[~fixed], [0, 1]).all()


def test_same_law_reaches_rod_patterns_and_fuel_inventory_extremes():
    cases = sampler(4096)
    report = cases.preview(4096)
    assert report["preview_variable_fuel2_count_range"] == [0, len(cases.free_fuel)]
    counts = np.array(report["preview_rod_count_histogram"])
    assert np.all(counts > 650) and np.all(counts < 1000)
    patterns = {cases.block_case(i)[1].tobytes() for i in range(4096)}
    assert len(patterns) == 16


def test_generation_is_prefix_stable_and_order_independent():
    a, b = sampler(20), sampler(100)
    for i in [19, 3, 0, 18]:
        assert configuration_hash(a[i]) == configuration_hash(b[i])
        assert configuration_hash(a[i]) == configuration_hash(a[i])


def test_full_core_has_independent_asymmetric_rods_and_correct_center_assembly():
    cases = sampler(100, "full")
    assert len(cases.rod_sites) == 9
    assert cases.base.shape == (17, 17)
    assert cases[0]["material_map"].shape == (136, 136)
    assert set(cases.config["bc"].values()) == {"vacuum"}
    assert all(np.array_equal(cases.base, np.flip(cases.base, axis)) for axis in (0, 1))
    assert any(
        not np.array_equal(cases.block_case(i)[1], np.fliplr(cases.block_case(i)[1]))
        for i in range(20)
    )
    # The central full assembly is eight cells wide (20 cm), not two copies
    # of a full-width assembly on the former quarter-core symmetry line.
    assert np.count_nonzero(cases.pixel_blocks == 8) == 8


def test_duplicate_configurations_cannot_leak_between_splits():
    cases = sampler(1000)
    hashes = [configuration_hash(cases[i]) for i in range(1000)]
    hashes.extend(hashes[:50])
    splits = grouped_splits(hashes, cases.seed)
    groups = [{hashes[i] for i in split} for split in splits.values()]
    assert sorted(i for split in splits.values() for i in split) == list(range(len(hashes)))
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])


def test_preview_cli_never_calls_a_solver(tmp_path, monkeypatch):
    import ndiffusion as nd

    def forbidden(*args, **kwargs):
        raise AssertionError("Preview must not start a physics solve")

    monkeypatch.setattr(nd, "KEigenSolverUnstructured2D", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["preview", "--config", str(CONFIG), "--out", str(tmp_path), "--n", "25"]
    )
    main()
    report = json.loads((tmp_path / "sampling_plan.json").read_text())
    assert report["physics_solves_performed"] == 0
    assert report["planned_samples"] == 25
    assert not (tmp_path / "samples.h5").exists()
