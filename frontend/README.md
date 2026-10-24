# Frontend

Scientist UI. Stack TBD (e.g. React + Vite, or Streamlit for speed).

Talks to the backend:
- `POST /predict` — surrogate prediction for a core layout
- `POST /solve` — reference ndiffusion solve for comparison
- `GET /health`

Must show: editable core layout grid, flux/power heatmaps with units and normalization stated,
`k_eff`, peaking factors, surrogate-vs-reference error map, out-of-distribution warning, and a
"research surrogate — not safety grade" label. Export layout + results as JSON/CSV.
