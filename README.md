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
bash scripts/get_dataset.sh data/released_5000
unzip data/released_5000/twinreactor-material-boron-5000.zip -d data/released_5000
python -m model.check_data --data data/released_5000/material_boron_5000 --models --report data/validation.json
pytest
```

For the full-field dataset, configurable material/rod inputs, and FNO/CNN/PINN
adapters, see [the data handoff](docs/DATA_HANDOFF.md). The generation command is
working; the original training and evaluation entry points remain placeholders.
The [Claude next-steps guide](docs/CLAUDE_DATA_NEXT_STEPS.md) identifies the remaining
training, normalization, evaluation and runtime-agent integration work. Generation:

```bash
python -m datagen.tabulated_cases --config datagen/configs/material_boron_5000.yaml --out data/new_5000
```
