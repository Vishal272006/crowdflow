"""
api.py

FastAPI backend for CrowdFlow. Wraps the trained LSTM (Phase 2) and RL
agent (Phase 3) behind three endpoints:

    POST /predict-bunching   -> LSTM forecast for the next 3 stops
    GET  /route-status       -> static route metadata (stops, hotspots)

Run with:
    uvicorn api:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for interactive API testing.
"""

import os
import sys
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import get_engine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
from route_model import ROUTE_CONFIG  # noqa: E402

app = FastAPI(title="CrowdFlow API", description="Bus bunching prediction and prevention for MTC Route 21G")

EXPECTED_WINDOW_SIZE = 5  # must match the window size the LSTM was trained with (prepare_sequences.py)

STOP_NAMES = ROUTE_CONFIG["stop_names"]
CONGESTION_INDEX = ROUTE_CONFIG["segment_congestion_index"]


class StopEvent(BaseModel):
    headway_deviation_min: float
    n_boarding: int
    dwell_sec: float
    hour_of_day: float
    headway_scheduled_min: float


class PredictRequest(BaseModel):
    window: List[StopEvent]  # most recent 5 stop-events, oldest first


@app.on_event("startup")
def load_models():
    get_engine()  # warm the singleton so first request isn't slow


@app.get("/")
def root():
    return {"service": "CrowdFlow API", "status": "running"}


@app.post("/predict-bunching")
def predict_bunching(req: PredictRequest):
    if len(req.window) != EXPECTED_WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"'window' must contain exactly {EXPECTED_WINDOW_SIZE} stop-events "
                   f"(oldest first) -- the LSTM was trained on {EXPECTED_WINDOW_SIZE}-stop "
                   f"history windows. Got {len(req.window)}."
        )
    engine = get_engine()
    window = [row.dict() for row in req.window]
    return engine.predict_bunching(window)


@app.get("/route-status")
def route_status():
    return {
        "route_id": ROUTE_CONFIG["route_id"],
        "stops": [
            {"idx": i, "name": name, "congestion_index": CONGESTION_INDEX[i] if i < len(CONGESTION_INDEX) else None}
            for i, name in enumerate(STOP_NAMES)
        ],
    }
