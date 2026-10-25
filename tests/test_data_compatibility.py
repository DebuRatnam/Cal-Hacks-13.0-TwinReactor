"""Check that configuration information survives solving, storage and loading."""

import json
import sys

import h5py
import ndiffusion as nd
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from datagen.ndiffusion_gen import main, solve_case
from datagen.physics_contract import QUARTER_CORE_BC, XS_FIELDS, resolve_xs
from datagen.schema import DatasetMetadata, write_dataset
from model.dataset import CoreDataset

ALL_REFLECTIVE = {side: "reflective" for side in QUARTER_CORE_BC}


@pytest.fixture
def table():
    m = nd.materials.IAEA.materials()
    return np.column_stack(
        [
            np.asarray(m.D).reshape(4, 2),
            np.asarray(m.removal).reshape(4, 2),
            np.asarray(m.nusigf).reshape(4, 2),
            np.asarray(m.scatter).reshape(4, 2, 2)[:, 1, 0],
        ]
    )


@pytest.fixture
def stored(tmp_path, table):
    layout = np.zeros((6, 8), dtype=int)
    samples = []
    for base, rod in ((table[0:1], 0.0), (table[1:2], 0.0), (table[1:2], 0.5)):
        s = solve_case(
            layout,
            {
                "xs_table": base,
                "rodded_xs_table": table[2:3],
                "rod_state": np.full(layout.shape, rod),
                "pitch_cm": 2.5,
                "bc": ALL_REFLECTIVE,
                "boundary_gamma": 0.4692,
            },
        )
        samples.append(s)
    meta = DatasetMetadata(
        "test-v1",
        "ndiffusion",
        nd.__version__,
        "IAEA",
        2,
        layout.shape,
        2.5,
        0,
        bc=ALL_REFLECTIVE,
        boundary_gamma=0.4692,
        flux_normalization="unit_integral_total_2group",
    )
    write_dataset(tmp_path, meta, samples)
    (tmp_path / "splits.json").write_text(json.dumps({"train": [2, 0], "val": [], "test": [1]}))
    return tmp_path, samples


def test_properties_and_partial_rods_are_visible_with_unchanged_ids(stored):
    path, samples = stored
    ds = CoreDataset(path, [2, 0, 1])
    assert ds.in_channels == len(XS_FIELDS) == 7
    assert torch.equal(ds[0]["material_map"], ds[1]["material_map"])
    assert not torch.equal(ds[0]["x"], ds[1]["x"])
    assert not torch.equal(ds[1]["x"], ds[2]["x"])
    assert np.isclose(float(ds[0]["k"]), samples[2].k_eff - 1, atol=1e-7)
    batch = next(iter(DataLoader(ds, batch_size=3)))
    assert batch["x"].shape == (3, 7, 6, 8)
    assert batch["flux"].shape == (3, 2, 6, 8)
    assert batch["power"].shape == (3, 6, 8)
    assert ds.physics_kwargs(batch)["gamma"] == 0.4692
    assert ds.physics_kwargs(batch)["h_cm"] == 2.5
    assert CoreDataset.from_split(path, "train").indices.tolist() == [2, 0]


def test_homogeneous_reference_matches_analytic_infinite_medium(stored):
    _, samples = stored
    for s in samples:
        _D1, _D2, sr1, sr2, nf1, nf2, s12 = s.xs[:, 0, 0]
        expected = (nf1 + nf2 * s12 / sr2) / sr1
        assert np.isclose(s.k_eff, expected, rtol=1e-7)
        assert s.solver_info["balance_residual"] < 1e-6


def test_pinn_reference_preserves_properties_and_group_ratio(stored):
    path, samples = stored
    case = CoreDataset(path, [2]).pinn_case(0)
    ref = case["reference"]
    mask = ref["mask"]
    np.testing.assert_allclose(
        ref["flux_grid"] * ref["source_normalization_scale"], samples[2].flux
    )
    source = samples[2].xs[4] * ref["flux_grid"][0] + samples[2].xs[5] * ref["flux_grid"][1]
    assert np.isclose(source[mask].mean(), 1, rtol=1e-12)
    assert case["pitch_cm"] == 2.5 and case["gamma"] == 0.4692
    assert ref["coordinates_cm"].shape == (48, 2)
    assert ref["k_eff"] == samples[2].k_eff


def test_mask_and_invalid_rods(table, tmp_path):
    ids = np.ones((6, 8), int)
    ids[-2:, -2:] = -1
    with pytest.raises(ValueError, match="endpoints"):
        resolve_xs(np.ones((6, 8), int), table, np.ones((6, 8)))
    rod = np.zeros(ids.shape)
    rod[0, 0] = 1
    s = solve_case(
        ids,
        {
            "xs_table": table,
            "rodded_xs_table": table,
            "rod_state": rod,
            "pitch_cm": 5.0,
            "boundary_gamma": 0.4692,
        },
    )
    meta = DatasetMetadata(
        "test", "ndiffusion", nd.__version__, "IAEA", 2, ids.shape, 5.0, 0, boundary_gamma=0.4692
    )
    write_dataset(tmp_path, meta, [s])
    ds = CoreDataset(tmp_path, [0], n_materials=4, features="onehot")
    item = ds[0]
    assert not item["mask"][-1, -1]
    assert torch.count_nonzero(item["x"][:, -2:, -2:]) == 0
    assert s.solver_info["balance_residual"] < 1e-6
    assert (s.power[~np.any(s.xs[4:6] > 0, axis=0)] == 0).all()


def test_explicit_config_cli_is_resumable_and_rejects_config_changes(table, tmp_path, monkeypatch):
    config = tmp_path / "cases.json"
    out = tmp_path / "out"
    cfg = {
        "version": "test",
        "seed": 0,
        "xs_source": "IAEA",
        "pitch_cm": 5.0,
        "bc": ALL_REFLECTIVE,
        "xs_table": table.tolist(),
        "cases": [{"material_map": np.ones((4, 4), int).tolist()}],
    }
    config.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["generator", "--config", str(config), "--out", str(out)])
    main()
    before = (out / "samples.h5").stat().st_mtime_ns
    main()
    assert (out / "samples.h5").stat().st_mtime_ns == before
    with h5py.File(out / "samples.h5") as f:
        assert f["xs"].shape == (1, 7, 4, 4)
    cfg["seed"] = 1
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="different generation config"):
        main()


def test_actual_architecture_batches_and_gradients(stored):
    pytest.importorskip("physicsnemo")
    from model import build_model
    from model.physics import physics_loss

    path, _ = stored
    ds = CoreDataset(path, [2, 0, 1])
    batch = next(iter(DataLoader(ds, batch_size=2)))
    for name, options in (
        ("fno", {"latent_channels": 8, "num_fno_layers": 1, "num_fno_modes": 3}),
        ("cnn", {"conv_layer_size": 8, "n_blocks": 1, "n_downsampling": 1}),
    ):
        net = build_model(name, in_channels=ds.in_channels, **options)
        pred = net(batch["x"])
        assert pred["flux"].shape == batch["flux"].shape
        assert pred["power"].shape == batch["power"].shape
        assert pred["k"].shape == batch["k"].shape
        sum(value.square().mean() for value in pred.values()).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    residual = physics_loss(batch["flux"], batch["k"] + 1, batch["xs"], **ds.physics_kwargs(batch))
    assert float(residual) < 1e-10
    problem, reference = ds.make_pinn_problem(0)
    pinn = problem.make_model(layer_size=16, num_layers=2)
    loss = problem.losses(pinn, n_interior=32, n_boundary=16)
    loss["total"].backward()
    assert all(torch.isfinite(value) for value in loss.values())
    assert pinn.log_k.grad is not None and torch.isfinite(pinn.log_k.grad)
    assert reference["flux_grid"].shape == (2, 6, 8)


def test_mesh_row_zero_is_south_and_column_zero_is_west(table):
    from datagen.ndiffusion_gen import build_mesh

    ids = np.zeros((3, 5), dtype=int)
    mask = np.ones_like(ids, dtype=bool)
    mask[-1, -1] = False
    h = 2.6775
    mesh, _, cells = build_mesh(ids, mask, h, table, QUARTER_CORE_BC, 0.4692)
    vx, vy = np.asarray(mesh.vx), np.asarray(mesh.vy)
    offsets, vertices = np.asarray(mesh.cell_offsets), np.asarray(mesh.cell_vertices)
    for cell, (row, column) in enumerate(cells):
        indices = vertices[offsets[cell] : offsets[cell + 1]]
        np.testing.assert_allclose(vx[indices].mean(), (column + 0.5) * h)
        np.testing.assert_allclose(vy[indices].mean(), (row + 0.5) * h)
    for a, b, tag in zip(mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag):
        south_or_west = (vy[a] == vy[b] == 0) or (vx[a] == vx[b] == 0)
        assert (tag == 0) == south_or_west


def test_asymmetric_reference_orientation_and_rayleigh_k(table):
    pytest.importorskip("physicsnemo")
    from model.physics import physics_loss, rayleigh_k

    ids = np.zeros((6, 9), dtype=int)
    ids[:, 4:] = 1
    ids[-2:, -3:] = -1
    rods = np.zeros_like(ids, dtype=float)
    rods[1:3, 1:3] = 1
    rodded = table.copy()
    rodded[0] = table[2]
    sample = solve_case(
        ids,
        {
            "xs_table": table,
            "rodded_xs_table": rodded,
            "rod_state": rods,
            "pitch_cm": 5.0,
            "bc": QUARTER_CORE_BC,
            "boundary_gamma": 0.4692,
        },
    )
    phi, xs, mask = (torch.from_numpy(getattr(sample, key))[None] for key in ("flux", "xs", "mask"))
    k = torch.tensor([sample.k_eff], dtype=torch.float64)
    kw = {"h_cm": 5.0, "bc": QUARTER_CORE_BC, "gamma": 0.4692, "void": ~mask}
    assert float(physics_loss(phi, k, xs, **kw)) < 1e-10
    assert float((rayleigh_k(phi, xs, **kw) - k).abs().max()) < 1e-6
    for axis in (-2, -1):
        flipped_kw = {**kw, "void": ~mask.flip(axis)}
        assert float(physics_loss(phi.flip(axis), k, xs.flip(axis), **flipped_kw)) > 1e-6
