"""Evaluate one-time departure control over fresh, simulated service days."""

import argparse
import json
import os
import numpy as np
from stable_baselines3 import PPO

from bus_env import BusDispatchEnv, DEPARTURE_DELAY_SEC


def no_delay_policy(_env, _observation):
    return 0


def fixed_rule_policy(_env, observation):
    """Delay 60 seconds only if the preceding bus is dangerously close."""
    return DEPARTURE_DELAY_SEC.index(60) if observation[0] < -0.5 else 0


def run_day(env, policy):
    observation, _ = env.reset()
    rewards, ahead, maximum_ahead, delays = [], [], [], []
    terminated = False
    while not terminated:
        action = policy(env, observation)
        if isinstance(action, tuple):
            observation, reward, terminated, _, info = env.depart_with_delay(action[1])
        else:
            observation, reward, terminated, _, info = env.step(action)
        rewards.append(reward)
        ahead.append(info["mean_ahead_ratio"])
        maximum_ahead.append(info["max_ahead_ratio"])
        delays.append(info["departure_delay_seconds"])
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_ahead_ratio": float(np.mean(ahead)),
        "pct_departures_high_risk": float(100 * np.mean(np.asarray(maximum_ahead) > 1.0)),
        "mean_departure_delay_seconds": float(np.mean(delays)),
        "pct_departures_delayed": float(100 * np.mean(np.asarray(delays) > 0)),
    }


def average(days):
    return {key: float(np.mean([day[key] for day in days])) for key in days[0]}


def main():
    parser = argparse.ArgumentParser(description="Evaluate departure-only control on fresh simulated days")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Optional RL model to include in the comparison")
    parser.add_argument("--n-days", type=int, default=100)
    parser.add_argument("--val-csv", type=str, default=None,
                        help="Retained for command compatibility; evaluation simulates fresh days directly")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="Retained for command compatibility; use --n-days for this controller")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--delay-cost-weight", type=float, default=0.15)
    parser.add_argument("--behind-weight", type=float, default=0.55,
                        help="Kept for command compatibility")
    args = parser.parse_args()

    rl_policy = None
    if args.model_path:
        model = PPO.load(args.model_path)

        def rl_policy(_env, observation):
            action, _ = model.predict(observation, deterministic=True)
            return int(action)

    def future_path_planner(env, _observation):
        return ("exact_delay", env.plan_current_departure()["recommended_departure_delay_seconds"])

    policies = [
        ("No departure delay", no_delay_policy),
        ("Fixed departure rule", fixed_rule_policy),
        ("Future-path departure planner", future_path_planner),
    ]
    if rl_policy is not None:
        policies.append(("RL departure controller", rl_policy))

    results = {}
    for name, policy in policies:
        # Same seed gives every policy the same sequence of simulated days.
        env = BusDispatchEnv(seed=args.seed,
                             delay_cost_weight=args.delay_cost_weight,
                             behind_weight=args.behind_weight)
        results[name] = average([run_day(env, policy) for _ in range(args.n_days)])

    print("=" * 100)
    print(f"{'Policy':<30}{'Reward':<10}{'Ahead |Dev|':<14}{'Departure Delay(s)':<20}{'% Departures Delayed'}")
    print("=" * 100)
    for name, result in results.items():
        print(f"{name:<30}{result['mean_reward']:<10.3f}{result['mean_ahead_ratio']:<14.3f}"
              f"{result['mean_departure_delay_seconds']:<20.1f}{result['pct_departures_delayed']:.1f}")

    print()
    print(f"{'Policy':<30}{'% departure decisions with High-Risk trip'}")
    print("-" * 70)
    for name, result in results.items():
        print(f"{name:<30}{result['pct_departures_high_risk']:.1f}")

    no_delay = results["No departure delay"]
    planner = results["Future-path departure planner"]
    improvement = 100 * (no_delay["mean_ahead_ratio"] - planner["mean_ahead_ratio"]) / no_delay["mean_ahead_ratio"]
    print(f"\nFuture-path planner vs no departure delay: ahead deviation improved {improvement:+.1f}%")

    if args.json_out:
        summary = {**results, "planner_ahead_improvement_pct_vs_no_delay": float(improvement)}
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w") as file:
            json.dump(summary, file, indent=2)
        print(f"JSON summary written to {args.json_out}")


if __name__ == "__main__":
    main()
