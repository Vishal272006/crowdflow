#!/usr/bin/env python3
"""
reward_sweep.py

Trains and evaluates the RL agent under several different reward-weight
configurations, to show the no-delay/fixed-rule/RL comparison isn't
arbitrary -- it visibly shifts depending on how holding cost and
bus-behind risk are weighted. This directly answers "why these weights
and not others?" in review. Every intervention happens once at departure.

Two sweeps:
  A. delay_cost_weight in {0.05, 0.15, 0.3}, behind_weight fixed at 1.0
     -- how much does penalizing a delayed departure change
     the agent's behavior?
  B. behind_weight in {0.0, 0.5, 1.0}, delay_cost_weight fixed at 0.15
     -- added after finding (see chat/report) that behind_weight=1.0 made
     the agent much MORE conservative than the original single-neighbor
     formulation (ahead-improvement dropped from ~28% to ~7%). This sweep
     shows exactly where that tradeoff comes from.

Usage:
    python reward_sweep.py
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-1500:])
        print(result.stderr[-1500:])
        raise RuntimeError("command failed")
    return result.stdout


def run_config(delay_cost_weight, behind_weight, tag, timesteps=150_000):
    out_dir = f"../data/reward_sweep/{tag}"
    print(f"\n--- Config: delay_cost_weight={delay_cost_weight}, behind_weight={behind_weight} ---")
    run([sys.executable, "train_agent.py", "--timesteps", str(timesteps),
         "--delay-cost-weight", str(delay_cost_weight), "--behind-weight", str(behind_weight),
         "--out-dir", out_dir])

    json_out = f"{out_dir}/eval.json"
    run([sys.executable, "evaluate_agent.py", "--n-episodes", "300",
         "--model-path", f"{out_dir}/best_model.zip",
         "--delay-cost-weight", str(delay_cost_weight), "--behind-weight", str(behind_weight),
         "--json-out", json_out])

    with open(os.path.join(ROOT, json_out)) as f:
        return json.load(f)


def main():
    results = {}

    print("=" * 70)
    print("SWEEP A: delay_cost_weight in {0.05, 0.15, 0.3}, behind_weight=1.0")
    print("=" * 70)
    for w in [0.05, 0.15, 0.3]:
        results[f"delay_cost_{w}"] = run_config(w, 1.0, f"delay_cost_{w}")

    print("\n" + "=" * 70)
    print("SWEEP B: behind_weight in {0.0, 0.5, 1.0}, delay_cost_weight=0.15")
    print("=" * 70)
    for w in [0.0, 0.5]:  # 1.0 already covered by hold_cost_0.15 above
        results[f"behind_{w}"] = run_config(0.15, w, f"behind_{w}")
    results["behind_1.0"] = results["delay_cost_0.15"]  # same config, reuse

    print("\n\n" + "=" * 70)
    print("SWEEP A RESULTS: varying delay_cost_weight (behind_weight=1.0)")
    print("=" * 70)
    print(f"{'delay_cost_weight':<20}{'Ahead |Dev|':<14}{'Behind |Dev|':<14}{'Departure Delay(s)':<20}{'Ahead Improv. vs No-Delay'}")
    for w in [0.05, 0.15, 0.3]:
        r = results[f"delay_cost_{w}"]["RL departure controller"]
        no_delay = results[f"delay_cost_{w}"]["No departure delay"]
        improv = 100 * (no_delay["mean_ahead_ratio"] - r["mean_ahead_ratio"]) / no_delay["mean_ahead_ratio"]
        print(f"{w:<20}{r['mean_ahead_ratio']:<14.3f}{r['mean_behind_ratio']:<14.3f}"
              f"{r['mean_departure_delay_seconds']:<20.1f}{improv:+.1f}%")

    print("\n" + "=" * 70)
    print("SWEEP B RESULTS: varying behind_weight (delay_cost_weight=0.15)")
    print("=" * 70)
    print(f"{'behind_weight':<20}{'Ahead |Dev|':<14}{'Behind |Dev|':<14}{'Departure Delay(s)':<20}{'Ahead Improv. vs No-Delay'}")
    for w in [0.0, 0.5, 1.0]:
        r = results[f"behind_{w}"]["RL departure controller"]
        no_delay = results[f"behind_{w}"]["No departure delay"]
        improv = 100 * (no_delay["mean_ahead_ratio"] - r["mean_ahead_ratio"]) / no_delay["mean_ahead_ratio"]
        print(f"{w:<20}{r['mean_ahead_ratio']:<14.3f}{r['mean_behind_ratio']:<14.3f}"
              f"{r['mean_departure_delay_seconds']:<20.1f}{improv:+.1f}%")

    with open(os.path.join(ROOT, "../data/reward_sweep/summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to ../data/reward_sweep/summary.json")


if __name__ == "__main__":
    main()
