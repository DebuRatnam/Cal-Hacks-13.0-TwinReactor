# Released 5,000-case data pipeline

The full dataset is in the repository's
[dataset-material-boron-5000-v3 release](https://github.com/DebuRatnam/Cal-Hacks-13.0-TwinReactor/releases/tag/dataset-material-boron-5000-v3).
All 14 parts contain **all inputs, full fast/thermal flux, relative power, raw k_eff,
metadata, saved splits, provenance and matching generation source**. The CSV is only a summary.

```bash
bash scripts/get_dataset.sh data/released_5000
unzip data/released_5000/twinreactor-material-boron-5000.zip -d data/released_5000
python -m model.check_data --data data/released_5000/material_boron_5000 --models --report data/validation.json
```

The helper requires GitHub CLI and shasum/sha256sum. It verifies all parts and the joined ZIP.
Keep generated data out of Git. The release HDF5 uses one sample per compressed chunk for
random training reads; repacking was verified to preserve every original array value.

## Sampling and scope

Version `v3-nrc-uox-boron-5000`, seed `20261029`, 5,000 independent draws under one law:

- Six UO2 states: 4.2% and 4.5% enrichment, each at 0.15, 17.5 and 37.5 GWd/tHM burnup.
- Boron uniformly chosen from 0, 500, 1,000, 1,500 and 2,000 ppm. Native table nodes are
  0/1,000/2,000; interpolate transport/reaction coefficients before converting D=1/(3 Sigma_tr).
- Per-case fuel-age proportions drawn from Dirichlet(0.3,0.3,0.3), then assigned to assemblies.
  Per-case 4.5%-fuel probability uniform in [0,1], then enrichment assigned to each fuel assembly.
- Inserted-bank count uniform from 0 through 8, then that many banks selected uniformly.
  Eight banks cover 17 quarter-core sites; each bank is fully in/out together. This is a 2D
  cross section, not partial axial insertion. Inputs reflect across quarter-core symmetry planes.
- Temperature is fixed: fuel 900 K, moderator density 0.71187 g/cm3 (580 K branch).

No k_eff-based filtering. All 5,000 converged with 5,000 unique resolved inputs. k_eff range
**0.7989350541–1.2124433826**: 1,958 subcritical / 3,042 supercritical. Saved train/validation/test
counts are **3,999 / 489 / 512**; preserve the existing assignments.

This is an adapted 2D UO2 configuration study using NRC/OECD HELIOS tables via pinned KOMODO
source, **not a validated reproduction of the mixed UO2/MOX benchmark or coverage of every PWR
configuration**. Raw tables, source commit and MIT notice are in `datagen/library/nrc_mox_helios/`.
Burnable absorbers follow the source-library assumptions. Assembly discontinuity factors are
omitted. No old IAEA axial-buckling term is added. Project Robin vacuum boundaries are retained
rather than the source benchmark's zero-flux radial condition. EECS review remains requested
for these physical approximations.

Do not concatenate this with the earlier IAEA 3,500-case data: the material catalog, domain and
cell width differ. Legacy IAEA export/sampling/merge utilities are retained, but that merge
helper does not support the v3 material/boron dataset.

## Stored arrays

| HDF5 key | Shape | Meaning |
|---|---|---|
| `xs` | `[N,7,68,68]` | Exact resolved solver coefficients |
| `material_map` | `[N,68,68]` | Semantic fuel/reflector IDs; -1 outside domain |
| `rod_state`, `mask` | `[N,68,68]` | Binary rod map; fuel+reflector domain mask |
| `flux` | `[N,2,68,68]` | Fast then thermal flux |
| `power` | `[N,68,68]` | Relative power, mean 1 over fuel |
| `k_eff` | `[N]` | Absolute k_eff |
| `boron_ppm` | `[N]` | Soluble boron input |
| `enrichment_wt`, `burnup_gwd_t` | `[N,68,68]` | Fuel-state descriptors |
| `kappa_fission` | `[N,2,68,68]` | Tabulated power coefficients |
| `assembly_material_map`, `assembly_rod_state` | `[N,9,9]` | Coarse inputs, including central half-width blocks |

`solver_info_json` records per-case convergence and independent discrete-balance residuals.
Coefficient channels are exactly:

```text
D1, D2, sig_r1, sig_r2, nu_sig_f1, nu_sig_f2, sig_s12
```

D and lengths use cm; reaction coefficients use cm^-1. Fast removal includes absorption plus
downscatter. No upscatter; chi=[1,0]. Do not add scatter or axial leakage again. A global
semantic-ID table is insufficient: boron and local rods change coefficients. Use `xs` directly.
For a table representation, `datagen.physics_contract.compact_xs(xs, mask)` returns exact
matching per-case physics IDs, a coefficient table with a zero padding row, and the void ID.

## Confirmed orientation and boundary contract

Arrays are `[row,column]=[y,x]`. Row 0 is the southernmost cell row, column 0 is west; rows
increase northward. Cell centers are `x=(column+0.5)*h`, `y=(row+0.5)*h`; the south/west faces,
not centers, are at zero. Plot with `imshow(..., origin="lower")`; no array flip or transpose.

The numerical cell width is **h=2.6775 cm**, separate from the **21.42 cm assembly pitch**.
West/south reflect. East/north and exposed stepped domain faces use
`D*dphi/dn + 0.4692*phi = 0`. Use saved metadata: model default gamma=0.5 stays generic for
other callers, including the existing placeholder agent.

Tests check coordinates/boundary tags on a non-square stepped mesh and PDE/Rayleigh agreement
on an asymmetric solve. Deliberate row/column flips with unchanged boundaries fail.
`MAIN_PHYSICS_VERIFICATION_5000.json` records the all-5,000-case check against main `6e3b60d`:
maximum per-case relative squared FV residual **1.51e-16**, maximum absolute Rayleigh k error
**1.46e-8**. This establishes compatibility with the equations solved, not transport accuracy.

## Model interfaces

```python
from pathlib import Path
from torch.utils.data import DataLoader
from model import build_model
from model.dataset import CoreDataset
from model.physics import physics_loss, rayleigh_k

ds = CoreDataset.from_split(Path("data/released_5000/material_boron_5000"), "train")
batch = next(iter(DataLoader(ds, batch_size=16, shuffle=True)))
net = build_model("fno", in_channels=ds.in_channels)  # "cnn" uses the same contract
pred = net(batch["x"])  # default x is seven resolved coefficient channels
loss_pde = physics_loss(pred["flux"], pred["k"] + 1, batch["xs"], **ds.physics_kwargs(batch))
k_from_flux = rayleigh_k(pred["flux"], batch["xs"], **ds.physics_kwargs(batch))
```

The loader accepts unsorted/duplicate indices without h5py fancy-index errors, reads one case
at a time, and masks void rather than indexing -1 into a one-hot table. `batch['k']` is k_eff-1;
HDF5 k is absolute. Keep `batch['xs']` in physical units even if `batch['x']` is standardized.
Flux uses a common two-group scale: `sum(flux)*h**2=1`. Undo target standardization before
physics checks; independent group scaling otherwise corrupts the physical group ratio.
Mask field losses with `batch['mask']`; use `batch['fuel_mask']` for fuel-power metrics.
Power is tabulated kappa-fission times flux, mean 1 over fuel, not absolute watts.

```python
problem, reference = ds.make_pinn_problem(0, interface_width_cm=0.5)
pinn = problem.make_model()
```

The PINN trains on **one configuration**, not an operator across layouts. Its adapter builds
the matching coefficient table and rescales both groups together to mean fission source=1.
Existing 0.5 cm interface smoothing is an explicit approximation. No final surrogate training
or PCA fit is included. Trainer/evaluator entry points remain stubs; see `CLAUDE_DATA_NEXT_STEPS.md`.

## Generate or audit

```bash
python -m datagen.tabulated_cases --config datagen/configs/material_boron_5000.yaml --out data/new_5000
python -m model.check_data --data data/released_5000/material_boron_5000 --models --report data/validation.json
pytest
```

Generation needs NumPy, SciPy, h5py, PyYAML and ndiffusion. Model audit needs the model extras.
Native ARM64 Python 3.14 and ndiffusion 1.1.0 were tested. The original 5,000 solves took 6,441
seconds on this Mac. Interrupted runs reuse NPZ shards with the same config/source fingerprint;
completed output directories are protected. Frozen original source remains in the release.
This commit adds metadata/storage/audit improvements, so do not resume frozen older shards
with the new code; use a new output directory or the frozen source.
