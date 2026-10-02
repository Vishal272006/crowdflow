"""
evaluate.py

Detailed evaluation of the trained CrowdFlow LSTM on the validation set:
  - Regression MAE per forecast horizon step (t+1, t+2, t+3), LSTM vs.
    persistence baseline
  - Classification precision/recall/F1 per risk class (Low/Med/High),
    LSTM vs. TWO baselines:
      1. "Always predict Low" (majority class) -- the weak floor
      2. PERSISTENCE: "assume the risk class stays whatever it is right
         now" -- a much fairer baseline than majority-class, since it
         actually uses the most recent real signal. If the LSTM can't
         beat this, it isn't learning temporal dynamics, just current
         state.
  - Confusion matrix
  - Writes a JSON summary (--json-out) for multi-seed aggregation

Usage:
    python evaluate.py
    python evaluate.py --seq-dir ../data/sequences --model-dir ../data/model --json-out ../data/model/eval_summary.json
"""

import argparse
import json
import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from lstm_model import CrowdFlowLSTM
from prepare_sequences import risk_class, FEATURE_COLS

CLASS_NAMES = ["Low", "Medium", "High"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dir", type=str, default="../data/sequences")
    parser.add_argument("--model-dir", type=str, default="../data/model")
    parser.add_argument("--json-out", type=str, default=None,
                         help="Optional path to write a JSON metrics summary")
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, "model_config.json")) as f:
        cfg = json.load(f)
    scaler = np.load(os.path.join(args.model_dir, "scaler.npz"))
    mean, std = scaler["mean"], scaler["std"]

    val_data = np.load(os.path.join(args.seq_dir, "val_sequences.npz"))
    X_val, y_reg_val, y_cls_val = val_data["X"], val_data["y_reg"], val_data["y_cls"]
    X_val_norm = (X_val - mean) / std

    device = torch.device("cpu")
    model = CrowdFlowLSTM(n_features=cfg["n_features"], hidden_size=cfg["hidden_size"],
                           horizon=cfg["horizon"])
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.pt"),
                                      map_location=device))
    model.eval()

    with torch.no_grad():
        reg_pred, cls_logits = model(torch.tensor(X_val_norm, dtype=torch.float32))
    reg_pred = reg_pred.numpy()
    cls_pred = cls_logits.argmax(dim=-1).numpy()

    # --- Regression: per-horizon-step MAE, LSTM vs persistence ---
    dev_col = FEATURE_COLS.index("headway_deviation_min")
    sched_col = FEATURE_COLS.index("headway_scheduled_min")
    naive_pred = X_val[:, -1, dev_col:dev_col + 1].repeat(cfg["horizon"], axis=1)

    print("=" * 65)
    print("REGRESSION: Mean Absolute Error (minutes) per horizon step")
    print("=" * 65)
    print(f"{'Horizon':<12}{'LSTM MAE':<14}{'Persist. MAE':<16}{'Improvement'}")
    reg_results = []
    for h in range(cfg["horizon"]):
        lstm_mae = float(np.mean(np.abs(reg_pred[:, h] - y_reg_val[:, h])))
        naive_mae = float(np.mean(np.abs(naive_pred[:, h] - y_reg_val[:, h])))
        improvement = 100 * (naive_mae - lstm_mae) / naive_mae
        print(f"t+{h+1:<10} {lstm_mae:<14.3f}{naive_mae:<16.3f}{improvement:+.1f}%")
        reg_results.append({"horizon": h + 1, "lstm_mae": lstm_mae,
                             "persistence_mae": naive_mae, "improvement_pct": improvement})

    # --- Classification: LSTM vs two baselines ---
    y_cls_flat = y_cls_val.flatten()
    cls_pred_flat = cls_pred.flatten()

    # Persistence baseline: current risk class held constant into the future.
    # Uses the SAME risk_class() thresholds prepare_sequences.py trained against.
    current_dev = X_val[:, -1, dev_col]
    current_sched = X_val[:, -1, sched_col]
    persistence_cls_single = np.array(
        [risk_class(d, s) for d, s in zip(current_dev, current_sched)], dtype=np.int64
    )
    persistence_cls_flat = np.repeat(persistence_cls_single[:, None], cfg["horizon"], axis=1).flatten()

    print()
    print("=" * 65)
    print("CLASSIFICATION: Bunching Risk (all horizon steps pooled)")
    print("=" * 65)
    print("-- LSTM --")
    print(classification_report(y_cls_flat, cls_pred_flat, target_names=CLASS_NAMES, digits=3))
    print("-- Persistence baseline (current risk class held flat) --")
    print(classification_report(y_cls_flat, persistence_cls_flat, target_names=CLASS_NAMES, digits=3))

    print("Confusion matrix -- LSTM (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_cls_flat, cls_pred_flat)
    header = "        " + "".join(f"{c:>10}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:<8}" + "".join(f"{v:>10}" for v in row))

    naive_cls_acc = float(np.mean(y_cls_flat == 0))
    persistence_cls_acc = float(np.mean(y_cls_flat == persistence_cls_flat))
    lstm_cls_acc = float(np.mean(y_cls_flat == cls_pred_flat))

    cm_persist = confusion_matrix(y_cls_flat, persistence_cls_flat, labels=[0, 1, 2])
    persist_high_recall = float(cm_persist[2, 2] / max(cm_persist[2].sum(), 1))
    lstm_high_recall = float(cm[2, 2] / max(cm[2].sum(), 1))

    print(f"\n{'Metric':<40}{'LSTM':<12}{'Persistence':<14}{'Always-Low'}")
    print(f"{'Overall accuracy':<40}{lstm_cls_acc:<12.3f}{persistence_cls_acc:<14.3f}{naive_cls_acc:.3f}")
    print(f"{'High-risk recall':<40}{lstm_high_recall:<12.3f}{persist_high_recall:<14.3f}{'--'}")
    print("\n(Always-Low is the WEAK floor -- any real model should crush it.")
    print(" Persistence is the FAIR baseline -- it already uses the latest real signal,")
    print(" so beating it is what actually proves the LSTM learned temporal dynamics.)")

    if args.json_out:
        summary = {
            "regression": reg_results,
            "classification": {
                "lstm_accuracy": lstm_cls_acc,
                "persistence_accuracy": persistence_cls_acc,
                "always_low_accuracy": naive_cls_acc,
                "lstm_high_risk_recall": lstm_high_recall,
                "persistence_high_risk_recall": persist_high_recall,
            },
        }
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nJSON summary written to {args.json_out}")


if __name__ == "__main__":
    main()
