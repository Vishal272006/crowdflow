"""
inference.py

Shared inference logic for CrowdFlow. Loads the trained LSTM (Phase 2) and
RL agent (Phase 3) once, and exposes two functions:

    predict_bunching(window)  -> LSTM forecast: deviation + risk class for
                                  the next 3 stops
    recommend_departure(state) -> RL agent's one-time departure delay

Both api.py (FastAPI backend) and the Streamlit dashboard import from this
file, so there's exactly one place that knows how to load and run the
models -- avoids the API and dashboard silently drifting out of sync.
"""

import json
import os
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))

from lstm_model import CrowdFlowLSTM  # noqa: E402
from route_model import ROUTE_CONFIG  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(DATA_DIR, "dr90", "model")
N_STOPS = len(ROUTE_CONFIG["stop_names"])

FEATURE_COLS = ["headway_deviation_min", "n_boarding", "dwell_sec",
                 "hour_of_day", "headway_scheduled_min"]
RISK_LABELS = ["Low", "Medium", "High"]


class CrowdFlowInference:
    """Loads both trained models once for prediction and departure control."""

    def __init__(self):
        with open(os.path.join(MODEL_DIR, "model_config.json")) as f:
            self.lstm_cfg = json.load(f)
        scaler = np.load(os.path.join(MODEL_DIR, "scaler.npz"))
        self.scaler_mean = scaler["mean"]
        self.scaler_std = scaler["std"]

        self.lstm = CrowdFlowLSTM(
            n_features=self.lstm_cfg["n_features"],
            hidden_size=self.lstm_cfg["hidden_size"],
            horizon=self.lstm_cfg["horizon"],
        )
        self.lstm.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, "best_model.pt"), map_location="cpu"))
        self.lstm.eval()


    def predict_bunching(self, window):
        """
        window: list of dicts, each with keys matching FEATURE_COLS, length
                = window size the LSTM was trained with (5 stops of history).
        Returns: {
            "predicted_deviation_min": [t+1, t+2, t+3],
            "risk_class": ["Low"/"Medium"/"High", ...],
        }
        """
        feats = np.array([[row[c] for c in FEATURE_COLS] for row in window], dtype=np.float32)
        feats_norm = (feats - self.scaler_mean) / self.scaler_std
        x = torch.tensor(feats_norm, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            reg_out, cls_out = self.lstm(x)

        deviations = reg_out.numpy()[0].tolist()
        risk_idx = cls_out.argmax(dim=-1).numpy()[0].tolist()
        risk_labels = [RISK_LABELS[i] for i in risk_idx]

        return {"predicted_deviation_min": deviations, "risk_class": risk_labels}

# module-level singleton -- models loaded once on first import
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = CrowdFlowInference()
    return _engine
