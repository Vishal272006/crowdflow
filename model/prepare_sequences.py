"""
prepare_sequences.py

Converts the flat per-stop-event CSV (train.csv / val.csv) into windowed
sequences suitable for LSTM training:

  Input:  past WINDOW_SIZE stop-events for one bus trip (features below)
  Output: headway_deviation_min (regression) AND bunching risk class
          (classification) for the NEXT HORIZON stops

Risk class thresholds match the ones already presented in the review deck
(Slide 7 -- "Dataset Framework"):
    Low    : |deviation| <= 0.5 * scheduled_headway
    Medium : 0.5 * scheduled_headway < |deviation| <= 1.0 * scheduled_headway
    High   : |deviation| > 1.0 * scheduled_headway

bus_id == 1 is EXCLUDED from every day: it's the first bus to serve every
stop that day, so headway_deviation_min is undefined (empty) for the whole
trip -- there's no previous bus to compare against. Every other bus (2..60)
has valid targets at every one of its 18 stops.

Usage:
    python prepare_sequences.py --window 5 --horizon 3
"""

import argparse
import numpy as np
import pandas as pd
import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
from route_model import ROUTE_CONFIG  # noqa: E402

FEATURE_COLS = [
    "headway_deviation_min",
    "n_boarding",
    "dwell_sec",
    "hour_of_day",
    "headway_scheduled_min",
]

N_STOPS = len(ROUTE_CONFIG["stop_names"])  # MTC Route 21G -- derived, not hardcoded


def risk_class(deviation, scheduled_headway):
    """0 = Low, 1 = Medium, 2 = High -- thresholds match the PPT deck."""
    ratio = abs(deviation) / max(scheduled_headway, 1e-6)
    if ratio <= 0.5:
        return 0
    elif ratio <= 1.0:
        return 1
    else:
        return 2


def build_sequences(df, window, horizon):
    """
    For each bus trip (day, bus_id) with bus_id != 1, slide a window across
    the 18 stops. Early positions are LEFT-PADDED with zeros (a common,
    simple choice -- the model learns to treat all-zero history as "start
    of trip", which is a real, recurring situation, not an edge case to
    hide).
    """
    df = df.sort_values(["day", "bus_id", "stop_idx"]).reset_index(drop=True)
    df = df[df["bus_id"] != 1].copy()  # bus_id==1 has no valid headway data

    X_list, y_reg_list, y_cls_list, meta_list = [], [], [], []

    for (day, bus_id), group in df.groupby(["day", "bus_id"]):
        group = group.sort_values("stop_idx").reset_index(drop=True)
        if len(group) != N_STOPS:
            continue  # skip incomplete trips (shouldn't happen, but be safe)

        feats = group[FEATURE_COLS].to_numpy(dtype=np.float32)
        sched_headway = group["headway_scheduled_min"].to_numpy(dtype=np.float32)
        deviation = group["headway_deviation_min"].to_numpy(dtype=np.float32)

        # valid current-stop indices i: need i+horizon <= N_STOPS-1
        for i in range(0, N_STOPS - horizon):
            start = i - window + 1
            if start < 0:
                pad = np.zeros((abs(start), len(FEATURE_COLS)), dtype=np.float32)
                window_feats = np.concatenate([pad, feats[0:i + 1]], axis=0)
            else:
                window_feats = feats[start:i + 1]

            future_dev = deviation[i + 1:i + 1 + horizon]
            future_sched = sched_headway[i + 1:i + 1 + horizon]
            future_cls = np.array(
                [risk_class(d, s) for d, s in zip(future_dev, future_sched)],
                dtype=np.int64
            )

            X_list.append(window_feats)
            y_reg_list.append(future_dev)
            y_cls_list.append(future_cls)
            meta_list.append((int(day), int(bus_id), int(i)))

    X = np.stack(X_list)               # (samples, window, n_features)
    y_reg = np.stack(y_reg_list)       # (samples, horizon)
    y_cls = np.stack(y_cls_list)       # (samples, horizon)
    return X, y_reg, y_cls, meta_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--out-dir", type=str, default="../data/sequences")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for split in ["train", "val"]:
        df = pd.read_csv(os.path.join(args.data_dir, f"{split}.csv"))
        X, y_reg, y_cls, meta = build_sequences(df, args.window, args.horizon)

        np.savez(
            os.path.join(args.out_dir, f"{split}_sequences.npz"),
            X=X, y_reg=y_reg, y_cls=y_cls
        )

        cls_counts = np.bincount(y_cls.flatten(), minlength=3)
        print(f"{split}: X{X.shape}  y_reg{y_reg.shape}  y_cls{y_cls.shape}  "
              f"risk class counts (Low/Med/High)={cls_counts.tolist()}")

    config = {"window": args.window, "horizon": args.horizon,
              "feature_cols": FEATURE_COLS, "n_stops": N_STOPS}
    with open(os.path.join(args.out_dir, "seq_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved sequence config -> {args.out_dir}/seq_config.json")


if __name__ == "__main__":
    main()
