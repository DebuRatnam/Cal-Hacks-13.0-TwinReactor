# PWR Neutronics Surrogate

Cal Hacks project: ML surrogate that predicts `k_eff`, flux, and power maps for 2-D PWR core
layouts, with a UI for research scientists. See [CLAUDE.md](CLAUDE.md) for design and conventions.

## Setup

```bash
conda env create -f environment.yml   # includes OpenMC
conda activate pwr-surrogate
export OPENMC_CROSS_SECTIONS=~/nucdata/endfb-viii.0-hdf5/cross_sections.xml
```

Without OpenMC: `pip install -e ".[model,backend,dev]"`.

## Workflow

```bash
python -m datagen.ndiffusion_gen --config datagen/configs/quarter_core_small.yaml --out data/v0
python -m model.train --data data/v0 --out checkpoints/v0
python -m model.evaluate --data data/v0 --ckpt checkpoints/v0
uvicorn backend.app:app --reload
pytest
```
