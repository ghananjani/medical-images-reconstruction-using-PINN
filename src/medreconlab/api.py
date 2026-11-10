"""Research-only synchronous API; job persistence/cancellation not yet supported."""
from __future__ import annotations
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .methods import METHODS
from .pipeline import experiment

app = FastAPI(title="MedReconLab", version="0.1.0",
              description="Synthetic research reconstruction. Not for clinical diagnosis.")
RUNS: dict[str, dict] = {}


class ReconstructionRequest(BaseModel):
    modality: str = "ct"
    size: int = Field(default=64, ge=16, le=256)
    seed: int = 0
    projections: int = Field(default=60, ge=2, le=720)
    acceleration: int = Field(default=4)
    noise: float = Field(default=0.01, ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok", "scope": "research-only"}


@app.get("/modalities")
def modalities():
    return list(METHODS)


@app.get("/methods")
def methods():
    return METHODS


@app.get("/datasets")
def datasets():
    return [{"name": "Shepp-Logan synthetic phantom", "available": True}]


@app.post("/reconstruct")
def reconstruct(payload: ReconstructionRequest):
    if payload.modality not in METHODS:
        raise HTTPException(422, "unsupported modality")
    if payload.acceleration not in (2, 4, 8):
        raise HTTPException(422, "acceleration must be 2, 4 or 8")
    try:
        outcome = experiment(**payload.model_dump())
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    job = str(uuid.uuid4())
    RUNS[job] = {"id": job, "status": "completed", "modality": payload.modality,
                 "parameters": outcome["parameters"],
                 "methods": {m: {"metrics": r["metrics"], "runtime_s": r["runtime_s"],
                                 "measurement_residual": r["measurement_residual"]}
                             for m, r in outcome["reconstructions"].items()}}
    return RUNS[job]


@app.post("/compare")
def compare(payload: ReconstructionRequest):
    return reconstruct(payload)


@app.get("/experiments")
def experiments():
    return list(RUNS.values())


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    if experiment_id not in RUNS:
        raise HTTPException(404, "experiment not found")
    return RUNS[experiment_id]


@app.get("/reports/{experiment_id}")
def report(experiment_id: str):
    return get_experiment(experiment_id)
