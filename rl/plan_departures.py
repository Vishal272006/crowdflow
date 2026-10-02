"""Run the explicit future-path planner for one simulated MTC 21G service day.

Each printed decision happens at Parrys Corner before a bus starts. The
planner projects Bus A's remaining trip and Bus B's full trip for every
allowed departure delay, then selects the lowest-risk option.
"""

import argparse

from bus_env import BusDispatchEnv


def main():
    parser = argparse.ArgumentParser(description="Show departure-only future-path decisions")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--delay-cost-weight", type=float, default=0.15)
    args = parser.parse_args()

    env = BusDispatchEnv(seed=args.seed, delay_cost_weight=args.delay_cost_weight)
    _, _ = env.reset()
    print("Bus 1 departs on schedule; planning begins for Bus 2.\n")

    terminated = False
    while not terminated:
        plan = env.plan_current_departure()
        best = plan["best_candidate"]
        current_bus = env.bus_ptr + 1
        decision = "DEPART NOW" if best["delay_seconds"] == 0 else f"WAIT {best['delay_seconds']} seconds"
        print(
            f"Bus {current_bus}: {decision} | closest projected gap: "
            f"{best['closest_gap_min']:.1f} min near {best['closest_stop_name']} "
            f"(scheduled gap: {best['scheduled_gap_min']:.1f} min)"
        )
        _, _, terminated, _, _ = env.depart_with_delay(best["delay_seconds"])


if __name__ == "__main__":
    main()
