"""Check the numeric library import and sampling without training a surrogate."""

from pathlib import Path

import numpy as np
import yaml

from datagen.physics_contract import resolve_xs
from datagen.tabulated_cases import LIBRARY, TabulatedCases, coefficients, interpolate, load_library


def cases():
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "datagen/configs/material_boron_100.yaml").read_text()
    )
    return TabulatedCases(cfg)


def test_known_numeric_node_matches_source_file():
    record = load_library(LIBRARY / "2G_XSEC_u42")[0.15][0]
    value = interpolate(record, 1000)
    np.testing.assert_allclose(
        value["Transport XSEC Table"], [0.235983, 0.834950], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        value["Absorption XSEC Table"], [0.0100398, 0.115282], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        value["Nu-Fission XSEC Table"], [0.00758079, 0.154909], rtol=0, atol=1e-12
    )
    assert np.isclose(value["Scattering XSEC Table"][1], 0.0162803)
    xs = coefficients(value)
    assert np.isclose(xs[2], 0.0100398 + 0.0162803)
    assert np.isclose(xs[0], 1 / (3 * 0.235983))


def test_interpolate_transport_before_converting_to_diffusion():
    record = load_library(LIBRARY / "2G_XSEC_u42")[0.15][0]
    left, right = interpolate(record, 0), interpolate(record, 1000)
    middle = interpolate(record, 500)
    np.testing.assert_allclose(
        middle["Transport XSEC Table"],
        (left["Transport XSEC Table"] + right["Transport XSEC Table"]) / 2,
    )
    expected_D = 1 / (3 * middle["Transport XSEC Table"])
    np.testing.assert_allclose(coefficients(middle)[:2], expected_D)


def test_all_material_boron_and_rod_states_are_valid():
    design = cases()
    for b in [0, 500, 1000, 1500, 2000]:
        tables, kappa = design.tables(b)
        assert tables.shape == (2, 7, 7) and kappa.shape == (2, 7, 2)
        for state in range(6):
            ids = design.base.copy()
            ids[design.fuel] = state
            sample = design.explicit(ids, (design.banks > 0).astype(float), b)
            xs = resolve_xs(
                sample["material_map"],
                sample["xs_table"],
                sample["rod_state"],
                sample["rodded_xs_table"],
            )
            assert np.isfinite(xs).all()
            assert not np.array_equal(tables[0, state], tables[1, state])


def test_banks_geometry_and_parameter_draws():
    design = cases()
    assert design.h == 2.6775
    assert np.count_nonzero(design.banks) == 17
    assert len(design.bank_ids) == 8
    seen_boron = set()
    seen_material = set()
    for i in range(100):
        case, row = design.case(i)
        again, _ = design.case(i)
        np.testing.assert_array_equal(case["material_map"], again["material_map"])
        np.testing.assert_array_equal(case["rod_state"], again["rod_state"])
        seen_boron.add(row["boron_ppm"])
        seen_material.update(case["assembly_material_map"][design.fuel].tolist())
        rods = case["assembly_rod_state"]
        assert np.all(rods[design.banks == 0] == 0)
        for bank in design.bank_ids:
            assert len(np.unique(rods[design.banks == bank])) == 1
    assert seen_boron == {0, 500, 1000, 1500, 2000}
    assert seen_material == set(range(6))
