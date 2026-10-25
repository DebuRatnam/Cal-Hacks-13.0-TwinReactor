# PWR Neutronics Surrogate

Cal Hacks project: ML surrogate that predicts `k_eff`, flux, and power maps for 2-D PWR core
layouts, with a UI for research scientists. See [CLAUDE.md](CLAUDE.md) for design and conventions.

## Open the website

Requires Python 3.11+ on macOS or Linux (on Windows, use WSL2). No GPU or dataset needed.

```bash
git clone https://github.com/DebuRatnam/Cal-Hacks-13.0-TwinReactor.git
cd Cal-Hacks-13.0-TwinReactor
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[model,backend,agent]"
export ANTHROPIC_API_KEY=...   # optional: only the NIA chat needs it
python -m uvicorn backend.app:app
```

Then open <http://127.0.0.1:8000>. The first install downloads PyTorch (several hundred MB).

## Development setup

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

For the full-field dataset, configurable material/rod inputs, and the FNO data
adapter, see [the data handoff](docs/DATA_HANDOFF.md). The surrogate is a PhysicsNeMo
FNO (`checkpoints/fno_softplus.pt`); train with `python -m model.train`, evaluate with
`python -m model.evaluate`.
The [Claude next-steps guide](docs/CLAUDE_DATA_NEXT_STEPS.md) identifies the remaining
training, normalization, evaluation and runtime-agent integration work. Generation:

```bash
python -m datagen.tabulated_cases --config datagen/configs/material_boron_5000.yaml --out data/new_5000
```
