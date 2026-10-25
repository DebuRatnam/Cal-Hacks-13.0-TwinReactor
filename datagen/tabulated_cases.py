"""UO2 material/burnup and boron states from the NRC/OECD HELIOS tables.

The dataset is an adapted 2-D UO2-only configuration study, not a reproduction
of the mixed UO2/MOX benchmark. Temperature/density are fixed, not sampled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import yaml

from datagen.iid_cases import grouped_splits
from datagen.ndiffusion_gen import solve_case
from datagen.physics_contract import QUARTER_CORE_BC
from datagen.schema import DatasetMetadata, Sample, write_dataset

LIBRARY = Path(__file__).with_name("library") / "nrc_mox_helios"


def numeric_lines(text):
    return [
        list(map(float, line.split()))
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("*")
    ]


def section(text, name):
    match = re.search(r"^\* " + re.escape(name) + r"\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing table {name}")
    tail = text[match.end() :]
    stop = re.search(r"^\* (?:.*Table|Fission Spectrum|Inverse Velocity)\s*$", tail, re.MULTILINE)
    if stop:
        tail = tail[: stop.start()]
    return np.asarray(numeric_lines(tail), dtype=float)


def load_library(path: Path):
    text = path.read_text()
    header = text.split("* BURNUP")[0]
    values = numeric_lines("\n".join(line.split("*")[0] for line in header.splitlines()))
    dims = tuple(map(int, values[1]))
    if dims not in ((3, 3, 3, 1), (1, 3, 1, 1)):
        raise ValueError(f"Unexpected library branch structure {dims}")
    if dims[0] == 3:
        np.testing.assert_allclose(values[2], [0.66114, 0.71187, 0.75206])
        np.testing.assert_allclose(values[3], [0, 1000, 2000])
        np.testing.assert_allclose(values[4], [560, 900, 1320])
    else:
        np.testing.assert_allclose(values[2], [0, 1000, 2000])
    burns = list(re.finditer(r"^\* BURNUP\s+([\d.]+)\s*$", text, re.MULTILINE))
    result = {}
    for i, match in enumerate(burns):
        chunk = text[match.end() : burns[i + 1].start() if i + 1 < len(burns) else len(text)]
        states = chunk.split("***RODDED XSEC*****")
        if len(states) != 2:
            raise ValueError("Each material/burnup needs matched unrodded and rodded tables")
        parsed = []
        for state in states:
            record = {}
            for name, groups in (
                ("Transport XSEC Table", 2),
                ("Absorption XSEC Table", 2),
                ("Nu-Fission XSEC Table", 2),
                ("Kappa-Fission XSEC Table", 2),
                ("Scattering XSEC Table", 4),
                ("ADF Table", 2),
            ):
                values = section(state, name)
                array = values.reshape(groups, dims[2], dims[1], dims[0])
                # Fixed fuel temperature 900 K and moderator density .71187
                # g/cm3 (580 K branch). Reflector tables have no such axes.
                record[name] = array[:, 1 if dims[2] > 1 else 0, :, 1 if dims[0] > 1 else 0]
            if np.any(record["Scattering XSEC Table"][2] != 0):
                raise ValueError("Current model contract does not support explicit upscatter")
            parsed.append(record)
        result[float(match.group(1))] = parsed
    return result


def interpolate(record, boron_ppm):
    if not 0 <= boron_ppm <= 2000:
        raise ValueError("Boron must remain inside the tabulated 0-2000 ppm range")
    return {
        key: np.array([np.interp(boron_ppm, [0, 1000, 2000], row) for row in value])
        for key, value in record.items()
    }


def coefficients(record):
    tr, sa, nf, scatter = (
        record[key]
        for key in (
            "Transport XSEC Table",
            "Absorption XSEC Table",
            "Nu-Fission XSEC Table",
            "Scattering XSEC Table",
        )
    )
    D = 1 / (3 * tr)
    return np.array([D[0], D[1], sa[0] + scatter[1], sa[1], nf[0], nf[1], scatter[1]])


class TabulatedCases:
    def __init__(self, config):
        self.config = config
        self.n, self.seed = int(config["n_samples"]), int(config["seed"])
        if self.n < 1 or self.seed < 0:
            raise ValueError("Use a positive sample count and nonnegative seed")
        if config["burnup_states_gwd_t"] != [0.15, 17.5, 37.5]:
            raise ValueError("This pilot uses the three explicit tabulated burnup states")
        if config["age_mix_concentration"] <= 0:
            raise ValueError("Age mix concentration must be positive")
        if not all(0 <= b <= 2000 for b in config["boron_states_ppm"]):
            raise ValueError("Boron choices must remain inside the tabulated range")
        self.states = [
            (enrichment, burnup)
            for enrichment in (4.2, 4.5)
            for burnup in config["burnup_states_gwd_t"]
        ]
        self.libraries = {
            4.2: load_library(LIBRARY / "2G_XSEC_u42"),
            4.5: load_library(LIBRARY / "2G_XSEC_u45"),
        }
        self.reflector = load_library(LIBRARY / "2G_XSEC_refl")[0.0]
        geo = json.loads((LIBRARY / "geometry.json").read_text())
        raw = np.asarray(geo["quarter_original_material_ids"])
        self.fuel = (raw > 0) & (raw != 18)
        self.base = np.where(raw == 0, -1, len(self.states)).astype(int)
        self.banks = np.asarray(geo["rod_banks"])
        self.bank_ids = np.unique(self.banks[self.banks > 0])
        self.h = geo["assembly_pitch_cm"] / 8
        edges = np.r_[0, 0.5, np.arange(1.5, 9)] * geo["assembly_pitch_cm"]
        self.blocks = np.searchsorted(edges, (np.arange(68) + 0.5) * self.h) - 1
        self.idx = np.ix_(self.blocks, self.blocks)
        self.fuel_sites = np.flatnonzero(self.fuel)

    def tables(self, boron):
        endpoints = [[], []]
        kappas = [[], []]
        for enrichment, burnup in self.states:
            for rod in (0, 1):
                rec = interpolate(self.libraries[enrichment][burnup][rod], boron)
                endpoints[rod].append(coefficients(rec))
                kappas[rod].append(rec["Kappa-Fission XSEC Table"])
        for rod in (0, 1):
            rec = interpolate(self.reflector[0], boron)
            endpoints[rod].append(coefficients(rec))
            kappas[rod].append(rec["Kappa-Fission XSEC Table"])
        return np.array(endpoints), np.array(kappas)

    def case(self, i):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, i]))
        boron = float(rng.choice(self.config["boron_states_ppm"]))
        age_weights = rng.dirichlet(np.full(3, self.config["age_mix_concentration"]))
        high_enrichment_fraction = float(rng.uniform())
        age = rng.choice(3, size=len(self.fuel_sites), p=age_weights)
        high = rng.random(len(self.fuel_sites)) < high_enrichment_fraction
        ids = self.base.copy()
        ids.flat[self.fuel_sites] = age + 3 * high
        n_banks = int(rng.integers(len(self.bank_ids) + 1))
        inserted = rng.choice(self.bank_ids, size=n_banks, replace=False)
        rod = np.isin(self.banks, inserted).astype(float)
        return self.explicit(ids, rod, boron), {
            "sample_id": i,
            "boron_ppm": boron,
            "inserted_bank_count": n_banks,
            "inserted_banks": inserted.tolist(),
            "age_mix_weights": age_weights.tolist(),
            "u45_probability": high_enrichment_fraction,
            "material_counts": np.bincount(ids[self.fuel], minlength=len(self.states)).tolist(),
        }

    def explicit(self, ids, rod, boron):
        table, kappa = self.tables(boron)
        return {
            "material_map": ids[self.idx],
            "rod_state": rod[self.idx],
            "xs_table": table[0],
            "rodded_xs_table": table[1],
            "pitch_cm": self.h,
            "bc": dict(QUARTER_CORE_BC),
            "boundary_gamma": 0.4692,
            "boron_ppm": boron,
            "kappa_tables": kappa,
            "assembly_material_map": ids,
            "assembly_rod_state": rod,
        }

    def solve(self, case):
        sample = solve_case(case["material_map"], case)
        mask = sample.mask
        ids = sample.material_map[mask]
        rods = sample.rod_state[mask].astype(int)
        kappa = np.zeros_like(sample.flux)
        kappa[:, mask] = case["kappa_tables"][rods, ids].T
        power = (kappa * sample.flux).sum(axis=0)
        fuel = mask & np.any(sample.xs[4:6] > 0, axis=0)
        sample.power = power / power[fuel].mean()
        return sample, kappa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    cases = TabulatedCases(config)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "samples.h5").exists():
        raise FileExistsError("Use a new dataset directory")
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
        + Path(__file__).read_bytes()
        + Path(__file__).with_name("ndiffusion_gen.py").read_bytes()
        + Path(__file__).with_name("physics_contract.py").read_bytes()
        + Path(__file__).with_name("schema.py").read_bytes()
        + b"".join(path.read_bytes() for path in sorted(LIBRARY.iterdir()) if path.is_file())
    ).hexdigest()
    marker = args.out / "generation.json"
    if marker.exists() and json.loads(marker.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Existing shards use a different config or source")
    marker.write_text(json.dumps({"fingerprint": fingerprint}, indent=2))
    shards = args.out / "shards"
    shards.mkdir(exist_ok=True)
    samples = []
    kappas = []
    records = []
    hashes = []
    maps = []
    bank_states = []
    for i in range(cases.n):
        case, record = cases.case(i)
        shard = shards / f"{i:06d}.npz"
        if shard.exists():
            with np.load(shard, allow_pickle=False) as z:
                sample = Sample(
                    **{
                        k: z[k].copy()
                        for k in (
                            "material_map",
                            "k_eff",
                            "flux",
                            "power",
                            "xs",
                            "rod_state",
                            "mask",
                        )
                    },
                    solver_info=json.loads(str(z["solver_info_json"])),
                )
                kappa = z["kappa_fission"].copy()
        else:
            sample, kappa = cases.solve(case)
            temp = shard.with_suffix(".tmp.npz")
            np.savez_compressed(
                temp,
                **{
                    k: getattr(sample, k)
                    for k in ("material_map", "k_eff", "flux", "power", "xs", "rod_state", "mask")
                },
                solver_info_json=json.dumps(sample.solver_info),
                kappa_fission=kappa,
            )
            temp.replace(shard)
        record.update(k_eff=sample.k_eff, seconds=sample.solver_info["seconds_total"])
        samples.append(sample)
        kappas.append(kappa)
        records.append(record)
        hashes.append(
            hashlib.sha256(
                sample.xs.tobytes() + sample.rod_state.tobytes() + sample.mask.tobytes()
            ).hexdigest()
        )
        maps.append(case["assembly_material_map"])
        bank_states.append(case["assembly_rod_state"])
        print(
            f"{i + 1}/{cases.n}: boron={record['boron_ppm']:.0f} ppm, banks={record['inserted_bank_count']}, k_eff={sample.k_eff:.7f}",
            flush=True,
        )
    import ndiffusion as nd

    names = [f"UO2_{enrichment}_burnup_{burnup}" for enrichment, burnup in cases.states] + [
        "reflector"
    ]
    meta = DatasetMetadata(
        config["version"],
        "ndiffusion",
        nd.__version__,
        "NRC/OECD MOX-UO2 HELIOS two-group library, UO2 subset; KOMODO b70d4ee numeric transcription",
        2,
        (68, 68),
        cases.h,
        cases.seed,
        material_names=names,
        boundary_gamma=0.4692,
        assembly_pitch_cm=21.42,
        flux_normalization="unit_integral_total_2group",
        scope="Adapted 2D UO2-only NRC-library study, fixed 900 K fuel and 580 K coolant; 0-2000 ppm interpolated boron. Plain diffusion without assembly discontinuity factors; not a reproduction of the mixed-MOX benchmark.",
    )
    (args.out / "inputs.json").write_text(json.dumps(config, indent=2))
    (args.out / "splits.json").write_text(json.dumps(grouped_splits(hashes, cases.seed), indent=2))
    (args.out / "input_hashes.json").write_text(json.dumps(hashes, indent=2))
    write_dataset(args.out, meta, samples)
    with h5py.File(args.out / "samples.h5", "a") as f:
        f["boron_ppm"] = np.array([r["boron_ppm"] for r in records])
        f.create_dataset("kappa_fission", data=np.stack(kappas), compression="gzip")
        f["assembly_material_map"] = np.stack(maps)
        f["assembly_rod_state"] = np.stack(bank_states)
        for name, values in (
            ("enrichment_wt", [a for a, b in cases.states] + [0]),
            ("burnup_gwd_t", [b for a, b in cases.states] + [0]),
        ):
            table = np.array(values)
            fields = np.stack(
                [
                    np.where(sample.mask, table[np.maximum(sample.material_map, 0)], 0)
                    for sample in samples
                ]
            )
            f.create_dataset(name, data=fields, compression="gzip")
    (args.out / "material_catalog.json").write_text(
        json.dumps(
            [
                {
                    "material_id": i,
                    "name": names[i],
                    "enrichment_wt": enrichment,
                    "burnup_gwd_t": burnup,
                }
                for i, (enrichment, burnup) in enumerate(cases.states)
            ]
            + [
                {
                    "material_id": len(cases.states),
                    "name": "reflector",
                    "enrichment_wt": 0,
                    "burnup_gwd_t": 0,
                }
            ],
            indent=2,
        )
    )
    (args.out / "samples.json").write_text(json.dumps(records, indent=2))
    fields = ["sample_id", "boron_ppm", "inserted_bank_count", "k_eff", "seconds"]
    with (args.out / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: r[key] for key in fields} for r in records)


if __name__ == "__main__":
    main()
