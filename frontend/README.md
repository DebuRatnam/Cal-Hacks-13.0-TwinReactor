# Frontend

ReactorTwin scientist UI: static page `index.html` (vanilla JS + three.js r128 from cdnjs), with
photos in `img/` (credits in the page footer). Started as the ReactorTwin claude.ai artifact; this
directory is now the source of truth.

- `site.js` — landing visuals (hero core twin, speed chart, photos, scroll) + shared helpers.
  The hero twin still shows illustrative sample data from the old stand-in solver.
- `console.js` — the console: loads the core from `GET /core`, edits layout / 8 rod banks /
  boron, calls `POST /predict` (FNO surrogate), shows assembly and 68×68 mesh maps,
  out-of-distribution / physics-check warnings, JSON export, and the agent chat (`POST /chat`).

Run (served by the backend, same origin as the API; the chat needs Claude API credentials):

    export ANTHROPIC_API_KEY=...
    uvicorn backend.app:app --reload
    open http://127.0.0.1:8000/

Talks to the backend:
- `GET /core` — geometry, materials, rod banks, units, training ranges, model test metrics
- `POST /predict` — surrogate prediction for a core layout
- `POST /chat` — agent turn; returns the reply, tool calls, and configurations it evaluated
- `GET /health`

`POST /solve` (reference ndiffusion) exists but is deliberately not in the UI for the MVP.
Not yet done: CSV export.

Must show: editable core layout grid, flux/power heatmaps with units and normalization stated,
`k_eff`, peaking factors, out-of-distribution warning, and a "research surrogate — not safety
grade" label. Export layout + results as JSON/CSV.
