"""FastAPI app.

Run:
    uvicorn backend.app:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="PWR Neutronics Surrogate")


class CoreLayout(BaseModel):
    material_map: list[list[int]]  # [H][W] indices into dataset material_names
    params: dict[str, float] = {}  # e.g. enrichment_wt, boron_ppm


class Prediction(BaseModel):
    k_eff: float
    flux: list[list[list[float]]]  # [G][H][W], normalized
    power: list[list[float]]  # [H][W], mean 1 over fuel cells
    peaking_factor: float
    out_of_distribution: bool
    source: str  # "surrogate" | "ndiffusion"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(layout: CoreLayout) -> Prediction:
    # TODO: load checkpoint once at startup, run model, OOD check against training ranges.
    raise HTTPException(501, "surrogate not implemented")


@app.post("/solve", response_model=Prediction)
def solve(layout: CoreLayout) -> Prediction:
    # TODO: call datagen.ndiffusion_gen.solve_case for the reference solution.
    raise HTTPException(501, "reference solver not implemented")
