# CLAUDE.md

## Current data handoff

Read [docs/CLAUDE_DATA_NEXT_STEPS.md](docs/CLAUDE_DATA_NEXT_STEPS.md) before changing
training, evaluation, or the runtime agent's material mapping. The released 5,000-case v3
dataset uses resolved coefficient inputs, not one-hot semantic material IDs. Its row-0-south
orientation is verified. The default runtime agent still uses a different placeholder IAEA
core and must not silently feed its inputs to a checkpoint trained on this release.

Cal Hacks project: ML surrogate for 2-D PWR core neutronics, with a UI for research scientists.

## Goal

Train a model that, given a 2-D PWR core layout (assembly/pin types, enrichment, burnable poison,
control rod state, boundary conditions), predicts:

- `k_eff` (scalar)
- multigroup scalar flux maps (2-group: fast + thermal)
- normalized power distribution and peaking factors (Fq, F_ΔH)

in milliseconds instead of seconds-to-hours for a full solve. Scientists use the UI to edit a core
layout, get instant predictions, and compare against a reference solver run.

## Team

- **EECS** — pipeline, model training infra, backend/API, reactor physics correctness
- **CS/Math** — model architecture, numerics, error analysis, solver convergence
- **MechE** — core geometry/materials, sanity-checking results

When a change touches physics assumptions (cross sections, geometry, boundary conditions,
normalization), flag it for the EECS teammate to review.

## Data generation

Two solvers. Use each for what it is good at:

| | NeutronDiffusion (`ndiffusion`) | OpenMC |
|---|---|---|
| Method | Deterministic multigroup diffusion (C++/pybind11) | Continuous-energy Monte Carlo transport |
| Speed | ms–s per 2-D solve | minutes+ per run, stochastic noise |
| Role | **Bulk training data** (10k+ samples) | **High-fidelity validation set** + generating few-group XS |
| Install | `pip install ndiffusion` | `conda install -c conda-forge openmc` (or Docker `openmc/openmc`) |

Default plan: train on `ndiffusion` data; use OpenMC to (a) homogenize assemblies into 2-group
cross sections fed to `ndiffusion`, and (b) build a small held-out set to measure the
diffusion-vs-transport gap. Do not mix the two sources in one training set without a label column.

### ndiffusion notes

- Docs: https://bwhewe-13.github.io/NeutronDiffusion/ — read the docs for exact API; don't guess signatures.
- Core objects: `nd.Materials` (fields `D`, `removal`, `scatter`, `chi`, `nusigf`, `n_groups`, `n_mat`),
  `nd.KEigenSolver(mats, medium_map, edges_x, [edges_y], geom, ...)`.
- Has built-in 2-group benchmarks (IAEA, BIBLIS, TWIGL, Ringhals) and preset Cartesian cores with
  symmetry sectors. **Use the IAEA/BIBLIS benchmarks as the first regression test** for the pipeline:
  reproduce published `k_eff` before generating any training data.
- Post-processing (power normalization, peaking factors) is built in — use it rather than reimplementing.

### OpenMC notes

- Cross-section library (ENDF/B-VIII.0 HDF5) is already on this machine:
  `export OPENMC_CROSS_SECTIONS=~/nucdata/endfb-viii.0-hdf5/cross_sections.xml`
  Depletion chains: `~/nucdata/chain_endfb80_thermal.xml`.
- 2-D = 3-D model with reflective boundaries in z.
- Always report `k_eff` with its standard deviation. Check Shannon entropy for source convergence
  before trusting tallies; discard enough inactive batches.
- Keep particle counts small during development; only scale up for final validation runs.

### Sampling the design space

Parameters to vary per sample (start small, widen later):
- Assembly layout on a quarter-core grid (exploit symmetry), e.g. 3–4 enrichment types + reflector
- Enrichment (~1.6–5.0 wt% U-235), soluble boron, burnable poison presence, control rod insertion
- Fuel/moderator temperature (only if the XS pipeline supports it)

Use Latin hypercube or Sobol sampling for continuous params, random valid layouts for discrete ones.
Reject invalid coefficients or failed/nonfinite solves and log why. The released v3
sampling law keeps all converged input draws; do not filter or rebalance it by `k_eff`.

## Data format

- One dataset = one directory with `samples.h5` (or `.npz` shards) + `metadata.json`
  (solver, solver version, XS source, mesh size, group structure, git commit, sampling seed).
- Per sample: input material map `[H, W]` int + per-material params, outputs `k_eff`,
  `flux [G, H, W]`, `power [H, W]` normalized to mean 1 over fuel cells.
- Fixed train/val/test split by seed, saved to disk. Never regenerate splits silently.
- Large data files are gitignored; commit only the generation scripts + configs.

## Model

- Framework: NVIDIA PhysicsNeMo (add as a dependency when ML work starts). Prefer its built-in
  architectures (e.g. FNO, physics-informed losses) over hand-rolled ones.
- Inputs are grid-structured → start with a U-Net or Fourier Neural Operator (FNO); a small CNN
  head regresses `k_eff`. `model/unet.py` is a shape-only placeholder.
- Normalize targets (flux per-group, `k_eff` as Δ from 1.0). Store normalization stats with the checkpoint.
- Metrics to report: `k_eff` error in **pcm** (1 pcm = 1e-5), max/RMS relative power error,
  peaking factor error. Target: |Δk| < 100 pcm, max power error < ~2–5%.
- Physics checks on predictions: flux ≥ 0, symmetric inputs → symmetric outputs, power zero in reflector.
- Always compare against a trivial baseline (e.g. mean prediction / nearest neighbor) in results.

## UI

For research scientists, not the general public. Priorities: correctness, units, and transparency.

- Core layout editor (click assemblies to change type/enrichment), instant surrogate prediction,
  heatmaps of flux/power, `k_eff` and peaking factors.
- Reference solver (`ndiffusion`) is not in the UI (decluttered for the MVP); it stays available
  via `POST /solve` and the agent's `verify_with_reference` tool. Report accuracy from the test set.
- Agent chat (`POST /chat`) runs the same surrogate as the console, starting from the core on
  screen; configurations it evaluates are added to the console list.
- Show uncertainty or an out-of-distribution warning when inputs fall outside the training range.
- Every displayed quantity has units and normalization stated.
- Export: layout + results as JSON/CSV.

## Suggested layout (update as the repo takes shape)

```
datagen/     # ndiffusion + OpenMC generation scripts, sampling configs
data/        # generated datasets (gitignored)
model/       # architectures, training, evaluation
backend/     # API serving model + reference solver
frontend/    # scientist UI
notebooks/   # exploration only; anything reused moves into a module
tests/       # benchmark regression tests (IAEA/BIBLIS k_eff), data schema tests
```

## Conventions

- Python 3.11+ (check that OpenMC's conda build supports the chosen version), type hints, `ruff` for lint/format.
- Units: cm for length, cm⁻¹ for macroscopic XS, eV for energy. Name variables with units when ambiguous (`pitch_cm`).
- Seed every random process; log seeds in dataset metadata.
- Long runs (data generation, training) must be resumable and write progress to disk.
- Hackathon pace: prefer working end-to-end slice first (tiny dataset → tiny model → UI shows a
  prediction), then scale each piece.

## Don'ts

- Don't present surrogate predictions as safety-grade results; label UI output as a research surrogate.
- Don't tune hyperparameters on the test set.
- Don't change group structure, mesh resolution, or normalization without regenerating data and
  bumping the dataset version.
