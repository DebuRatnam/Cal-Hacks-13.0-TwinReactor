# Claude coding-agent handoff

Start with `DATA_HANDOFF.md`, the current commit message (`git show -s --format=%B HEAD`),
and the files below. The released dataset is complete. This commit integrates generation,
coefficient-aware loading, per-case PINN inputs, validation and download helpers onto main.
It deliberately does not implement a trainer, train a final surrogate, or connect a checkpoint
to the UI. Keep the hackathon scope: fixed-temperature, 2D, two-group UO2 diffusion.

## First establish the working slice

1. Install the project and model/dev extras (`pip install -e '.[model,dev]'`). Install agent
   extras only if exercising the runtime agent. Download with
   `bash scripts/get_dataset.sh data/released_5000`, then extract the reconstructed ZIP there.
2. Run `python -m model.check_data --data data/released_5000/material_boron_5000 --models
   --report data/validation.json` and `pytest`. The audit checks all fields, reference PDE
   residuals and Rayleigh k, plus small FNO/CNN/PINN forward/backward interface checks.
   On a sandboxed Mac, point `WARP_CACHE_PATH` at a writable directory if needed.
3. Use `CoreDataset.from_split(..., 'train')` and `build_model(..., in_channels=ds.in_channels)`.
   Seven resolved coefficient channels are the default. Do not restore one-hot-only inputs.
   Preserve the 3,999/489/512 saved splits and original sample indices.

## Remaining implementation gaps, in priority order

1. **Training (`model/train.py`) is a stub.** Implement a small supervised FNO/CNN loop first.
   Fit input/target scaling only on the saved training split. Preserve physical `batch['xs']`
   and invert target scaling before `physics_loss`/`rayleigh_k`. Use `pred['k']+1` there.
   Use mask-aware losses, with fuel masks for power metrics. Flux and power have very
   different scales, so normalize or weight their losses deliberately. Save architecture
   settings, scaler state, dataset version/channel order, geometry, seed and optimizer state
   with each checkpoint; support resume. Do not optimize or choose a checkpoint on test data.
2. **Evaluation (`model/evaluate.py`) is a stub.** Compare with a train-only mean/nearest
   baseline. Report k error in pcm, masked flux and fuel-power errors, Rayleigh consistency,
   positivity and measured latency. Report subcritical/near-critical/supercritical performance
   separately. Existing `power_rel_error` assumes a shared spatial mask; adapt carefully if
   batching samples with different fuel masks. Use validation for model selection, test once.
3. **Per-case PINN:** keep it as a few representative physics baselines. Do not present
   `PINNProblem` as a general surrogate over layouts. Quantify sensitivity to the 0.5 cm
   smoothed interfaces and enforce compatible source normalization/boundaries. A successful
   gradient smoke test is not a trained-accuracy result.
4. **Runtime agent/configuration mapping:** `agent/core.py` currently constructs an IAEA
   placeholder 9x9 core, 21.5 cm pitch, four banks, axial buckling and an additive thermal
   rod proxy. `agent/fv_solver.py` uses gamma=0.5 and a rectangular domain without this
   release's stepped void mask; its power is a fission-rate proxy. Those are different physics
   from this 68x68 NRC-derived dataset (21.42 cm assemblies, h=2.6775, eight banks,
   gamma=0.4692, no added axial leakage, tabulated kappa-fission power). Preserve the existing
   placeholder backend for its current tests, but make that distinction explicit. Before
   adding `SurrogateBackend`, build one shared configuration-to-coefficient mapping using
   `datagen.tabulated_cases.TabulatedCases`/the pinned library and the actual quarter-core
   geometry. Do not blindly resize 9x9 inputs, reuse old material IDs, or attach the new
   checkpoint to the placeholder default. Extend the backend contract to carry mask,
   geometry, boundary settings and power conventions; route reference/surrogate through
   the same configuration. The FastAPI `/predict` and `/solve` endpoints are still stubs.
5. **Generation scaling:** the generator is resumable per case, but final assembly still
   holds solved fields in RAM. Stream from shards to HDF5 if scaling beyond comfortable
   memory. Keep one sample per compression chunk. Do not change the frozen release's
   arrays, splits or source fingerprint; use a new version/output directory for new sources.

## Physics invariants to preserve

- `xs` order: D1,D2,sig_r1,sig_r2,nu_sig_f1,nu_sig_f2,sig_s12. Removal already includes
  downscatter; no extra axial buckling in the v3 data; chi=[1,0], no upscatter.
- Confirmed `[row,column]=[y,x]`, row 0 south and column 0 west. Cell centers are
  ((column+0.5)*h,(row+0.5)*h). Plot with origin='lower'; no transpose/flip for physics.
- h=2.6775 cm, not 21.42 cm. Read `ds.physics_kwargs(batch)` for h/bcs/gamma/void.
- Raw HDF5 k is absolute; loader/model k is delta from 1. Flux groups share a common
  area-integral normalization; PINN references instead share a mean-source normalization.
- Full v3 data are in the release's 14 verified archive parts. The standalone CSV lacks
  layouts and fields. Do not merge with older IAEA data or fit PCA on held-out splits.

EECS review remains requested for the source-library adaptation, omitted discontinuity
factors, retained Robin boundary convention and PINN interface smoothing. The orientation
question itself is resolved by mesh-coordinate and asymmetric-solve tests. Preserve the
partner's model architectures and runtime-agent work unless a specific integration requires
a documented change.
