#!/usr/bin/env python3
"""
multi_seed_eval.py

Runs the ENTIRE pipeline (Phase 1 -> 2 -> 3) end-to-end for several
different random seeds, then aggregates the headline metrics into a
mean +/- std summary. This exists because a single seed's numbers --
"85% High-risk recall", "31% bunching events" -- could be seed-lucky.
If the numbers hold up (low variance) across 3 independent seeds, that's
real evidence of a robust result, not a cherry-picked run.

Each seed gets its own isolated directory under ../data/multiseed/seed_<N>/
so runs don't overwrite each other or the "official" single-seed results
in ../data/.

Usage:
    python multi_seed_eval.py --seeds 42 123 7
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SIMULATOR_DIR = os.path.join(ROOT, "simulator")
MODEL_DIR = os.path.join(ROOT, "model")
RL_DIR = os.path.join(ROOT, "rl")


def run(cmd, cwd):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


def run_one_seed(seed, days, out_root):
    seed_dir = os.path.join(out_root, f"seed_{seed}")
    data_dir = os.path.join(seed_dir, "data")
    seq_dir = os.path.join(data_dir, "sequences")
    lstm_model_dir = os.path.join(data_dir, "model")
    rl_model_dir = os.path.join(data_dir, "rl_model")
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n{'=' * 70}\nSEED {seed}\n{'=' * 70}")

    print("[1/6] Generating synthetic dataset...")
    run([sys.executable, "generate_dataset.py", "--days", str(days), "--seed", str(seed),
         "--out", os.path.join(data_dir, "synthetic_route.csv")], cwd=SIMULATOR_DIR)

    print("[2/6] Splitting train/val...")
    run([sys.executable, "split_dataset.py",
         "--in", os.path.join(data_dir, "synthetic_route.csv"),
         "--train-out", os.path.join(data_dir, "train.csv"),
         "--val-out", os.path.join(data_dir, "val.csv")], cwd=SIMULATOR_DIR)

    print("[3/6] Preparing LSTM sequences...")
    run([sys.executable, "prepare_sequences.py", "--window", "5", "--horizon", "3",
         "--data-dir", data_dir, "--out-dir", seq_dir], cwd=MODEL_DIR)

    print("[4/6] Training LSTM...")
    run([sys.executable, "train.py", "--epochs", "30", "--batch-size", "128", "--lr", "1e-3",
         "--seq-dir", seq_dir, "--out-dir", lstm_model_dir], cwd=MODEL_DIR)

    print("[5/6] Evaluating LSTM...")
    lstm_json = os.path.join(data_dir, "lstm_eval.json")
    run([sys.executable, "evaluate.py", "--seq-dir", seq_dir, "--model-dir", lstm_model_dir,
         "--json-out", lstm_json], cwd=MODEL_DIR)

    print("[6/6] Training + evaluating RL agent...")
    run([sys.executable, "train_agent.py", "--timesteps", "150000",
         "--train-csv", os.path.join(data_dir, "train.csv"),
         "--val-csv", os.path.join(data_dir, "val.csv"),
         "--out-dir", rl_model_dir], cwd=RL_DIR)
    rl_json = os.path.join(data_dir, "rl_eval.json")
    run([sys.executable, "evaluate_agent.py", "--n-episodes", "300",
         "--val-csv", os.path.join(data_dir, "val.csv"),
         "--model-path", os.path.join(rl_model_dir, "best_model.zip"),
         "--json-out", rl_json], cwd=RL_DIR)

    with open(lstm_json) as f:
        lstm_results = json.load(f)
    with open(rl_json) as f:
        rl_results = json.load(f)

    return lstm_results, rl_results


def summarize(values):
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()), "values": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--out-root", type=str, default="data/multiseed")
    args = parser.parse_args()

    out_root = os.path.abspath(os.path.join(ROOT, args.out_root))
    all_lstm, all_rl = [], []

    for seed in args.seeds:
        lstm_res, rl_res = run_one_seed(seed, args.days, out_root)
        all_lstm.append(lstm_res)
        all_rl.append(rl_res)

    high_recalls = [r["classification"]["lstm_high_risk_recall"] for r in all_lstm]
    accuracies = [r["classification"]["lstm_accuracy"] for r in all_lstm]
    persist_high_recalls = [r["classification"]["persistence_high_risk_recall"] for r in all_lstm]
    rl_improvements = [r["rl_ahead_improvement_pct_vs_no_delay"] for r in all_rl]

    summary = {
        "seeds": args.seeds,
        "lstm_high_risk_recall": summarize(high_recalls),
        "lstm_overall_accuracy": summarize(accuracies),
        "persistence_high_risk_recall": summarize(persist_high_recalls),
        "rl_ahead_improvement_pct_vs_no_delay": summarize(rl_improvements),
    }

    print(f"\n\n{'=' * 70}")
    print(f"MULTI-SEED SUMMARY across seeds {args.seeds}")
    print(f"{'=' * 70}")
    print(f"{'Metric':<38}{'Mean':<10}{'Std':<10}{'Min':<10}{'Max'}")
    for name, key in [
        ("LSTM High-risk recall", "lstm_high_risk_recall"),
        ("LSTM overall accuracy", "lstm_overall_accuracy"),
        ("Persistence High-risk recall", "persistence_high_risk_recall"),
        ("RL ahead improvement vs no-delay (%)", "rl_ahead_improvement_pct_vs_no_delay"),
    ]:
        s = summary[key]
        print(f"{name:<38}{s['mean']:<10.3f}{s['std']:<10.3f}{s['min']:<10.3f}{s['max']:.3f}")

    out_path = os.path.join(out_root, "multiseed_summary.json")
    os.makedirs(out_root, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary written to {out_path}")


if __name__ == "__main__":
    main()
