"""
train_agent.py

Trains the RL departure-control agent using PPO from Stable-Baselines3
on the BusDispatchEnv defined in bus_env.py.

--delay-cost-weight and --behind-weight expose the two reward-shaping
constants for sweeping (see reward_sweep.py) -- these directly change what
the agent is trained to optimize, not just how it's evaluated afterward.

Usage:
    python train_agent.py --timesteps 200000
    python train_agent.py --timesteps 150000 --delay-cost-weight 0.3 --behind-weight 0.5
"""

import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from bus_env import BusDispatchEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--train-csv", type=str, default=None,
                         help="Retained for command compatibility; the controller simulates full days directly")
    parser.add_argument("--val-csv", type=str, default=None,
                         help="Retained for command compatibility; the controller simulates full days directly")
    parser.add_argument("--out-dir", type=str, default="../data/rl_dispatch_model")
    parser.add_argument("--delay-cost-weight", type=float, default=0.15,
                         help="Penalty weight on delaying departure at the terminus")
    parser.add_argument("--behind-weight", type=float, default=0.55,
                         help="Relative weight on spacing risk to the bus behind")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_env = make_vec_env(
        lambda: BusDispatchEnv(seed=42,
                               delay_cost_weight=args.delay_cost_weight,
                               behind_weight=args.behind_weight),
        n_envs=4)
    eval_env = BusDispatchEnv(seed=123,
                              delay_cost_weight=args.delay_cost_weight,
                              behind_weight=args.behind_weight)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.out_dir,
        log_path=args.out_dir,
        eval_freq=5000,
        n_eval_episodes=100,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        gamma=0.98,
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(os.path.join(args.out_dir, "final_model"))
    print(f"\nTraining complete. Best model saved to {args.out_dir}/best_model.zip "
          f"(delay_cost_weight={args.delay_cost_weight}, behind_weight={args.behind_weight})")


if __name__ == "__main__":
    main()
