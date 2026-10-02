"""
generate_dataset.py

Runs the full route simulation for one or more service days and writes out
a synthetic dataset of per-stop bus arrival events, including the headway
deviation signal that bunching prediction (Phase 2, LSTM) will be trained on.

Now calibrated against the REAL MTC Route 21G published schedule:
  - 18 real stops, Broadway -> Tambaram
  - real free-flow segment times (from the 5:45 AM trip offsets)
  - real, non-uniform dispatch times (60 actual daily departures)
  - traffic congestion calibrated to the stated 18 km/h peak speed

The bunching feedback loop is still modeled explicitly:
  a bus that is running late -> finds more waiting passengers at the next
  stop (more time elapsed since the previous bus served it) -> longer dwell
  -> even later -> smaller headway to the bus behind it -> that bus finds
  fewer passengers -> catches up -> bunching.

Note: "scheduled headway" per stop is now the REAL gap to the previous
bus's scheduled dispatch time, not a fixed constant -- this matters because
the real schedule has 5-8 min peak gaps and 20-43 min midday gaps, and
"bunching" only makes sense relative to whatever gap was actually scheduled.

Usage:
    python generate_dataset.py --days 30 --seed 42 --out ../data/synthetic_route.csv
"""

import argparse
import csv
import os
import numpy as np

from route_model import ROUTE_CONFIG, n_stops, segment_free_flow_seconds, \
    sample_boarding_passengers, dwell_time_seconds
from traffic_model import sample_segment_delay_factor, sample_dwell_noise, schedule_recovery_speed_multiplier

# ---------------------------------------------------------------------------
# DOMAIN RANDOMIZATION -- bounded, per-day variation around the calibrated
# values in ROUTE_CONFIG. This is NOT "throw away calibration" -- it's
# controlled variance layered on top of it, so the sim covers a realistic
# DISTRIBUTION of days rather than one fixed configuration. Bounds were
# chosen conservatively (roughly +-15-40% around calibrated values) to stay
# inside the damped, realistic regime we tuned earlier -- wider bounds risk
# reintroducing the runaway-bunching instability an earlier version had
# before schedule-recovery damping was added.
#
#   - segment_congestion_index: jittered +-15% per segment, per day
#     (models day-to-day traffic variation -- an ordinary Tuesday isn't
#     identical to an ordinary Wednesday even at the same chokepoint)
#   - incident_prob / incident_scale: varied per day within a modest range
#     (some days just have more/worse random incidents than others)
#   - passenger "burst days": ~10% of days get a 1.5-2.5x passenger surge
#     across the whole route (festival, exam period, public holiday travel)
# ---------------------------------------------------------------------------

BURST_DAY_PROBABILITY = 0.10
BURST_MULTIPLIER_RANGE = (1.5, 2.5)
CONGESTION_JITTER_RANGE = (0.85, 1.15)
INCIDENT_PROB_RANGE = (0.004, 0.008)
INCIDENT_SCALE_RANGE = (1.4, 1.9)


def sample_daily_randomization(config, rng):
    """Draws ONE set of per-day randomized parameters, used for every stop
    and every bus that day (so the "personality" of a day is consistent
    across its own timeline, not re-rolled every stop)."""
    n_segments = len(config["segment_congestion_index"])

    is_burst_day = rng.random() < BURST_DAY_PROBABILITY
    burst_multiplier = rng.uniform(*BURST_MULTIPLIER_RANGE) if is_burst_day else 1.0

    congestion_jitter = rng.uniform(*CONGESTION_JITTER_RANGE, size=n_segments)
    day_congestion_index = [
        base * jitter for base, jitter in zip(config["segment_congestion_index"], congestion_jitter)
    ]

    day_incident_prob = rng.uniform(*INCIDENT_PROB_RANGE)
    day_incident_scale = rng.uniform(*INCIDENT_SCALE_RANGE)

    return {
        "is_burst_day": is_burst_day,
        "burst_multiplier": burst_multiplier,
        "segment_congestion_index": day_congestion_index,
        "incident_prob": day_incident_prob,
        "incident_scale": day_incident_scale,
    }


def simulate_one_day(config, day_index, rng, domain_randomization=True):
    """
    Simulates one full service day for the route, dispatching buses at the
    REAL published departure times rather than an assumed uniform headway.
    Returns a list of event dicts, one per (bus, stop) arrival.
    """
    n_stop = n_stops(config)
    dispatch_times = config["real_dispatch_times_min"]

    if domain_randomization:
        day_params = sample_daily_randomization(config, rng)
    else:
        day_params = {
            "is_burst_day": False, "burst_multiplier": 1.0,
            "segment_congestion_index": config["segment_congestion_index"],
            "incident_prob": 0.015, "incident_scale": 2.0,
        }

    last_bus_arrival = [None] * n_stop
    events = []

    for bus_id, sched_dispatch in enumerate(dispatch_times, start=1):
        t = float(sched_dispatch)

        if bus_id == 1:
            sched_headway_for_bus = dispatch_times[1] - dispatch_times[0]
        else:
            sched_headway_for_bus = sched_dispatch - dispatch_times[bus_id - 2]

        for stop_idx in range(n_stop):
            # `t` is the moment the bus ARRIVES at this stop.  Headway must
            # be measured from arrival to arrival, before either bus spends
            # time boarding passengers.  Recording it after dwell made later
            # buses look artificially too close at nearly every stop.
            arrival_time = t
            hour_float = (arrival_time % 1440) / 60.0

            if last_bus_arrival[stop_idx] is None:
                elapsed_since_last = sched_headway_for_bus
            else:
                elapsed_since_last = arrival_time - last_bus_arrival[stop_idx]

            n_boarding = sample_boarding_passengers(stop_idx, elapsed_since_last, rng, config)
            n_boarding = n_boarding * day_params["burst_multiplier"]
            n_boarding = min(int(round(n_boarding)), config["bus_capacity"])

            dwell_sec = dwell_time_seconds(n_boarding, config) * sample_dwell_noise(rng)

            headway_actual = None
            headway_deviation = None
            if last_bus_arrival[stop_idx] is not None:
                headway_actual = arrival_time - last_bus_arrival[stop_idx]
                headway_deviation = headway_actual - sched_headway_for_bus

            events.append({
                "day": day_index,
                "route_id": config["route_id"],
                "bus_id": bus_id,
                "stop_idx": stop_idx,
                "stop_name": config["stop_names"][stop_idx],
                "hour_of_day": round(hour_float, 3),
                "arrival_time_min": round(arrival_time, 3),
                "scheduled_dispatch_min": sched_dispatch,
                "n_boarding": int(n_boarding),
                "dwell_sec": round(dwell_sec, 2),
                "headway_actual_min": round(headway_actual, 3) if headway_actual is not None else "",
                "headway_scheduled_min": round(sched_headway_for_bus, 3),
                "headway_deviation_min": round(headway_deviation, 3) if headway_deviation is not None else "",
                "is_burst_day": int(day_params["is_burst_day"]),
            })

            last_bus_arrival[stop_idx] = arrival_time

            # Passenger boarding happens after arrival and affects the time
            # at which the bus can travel to the next stop.
            t = arrival_time + dwell_sec / 60.0

            if stop_idx < n_stop - 1:
                recovery_mult = 1.0
                if headway_deviation is not None:
                    recovery_mult = schedule_recovery_speed_multiplier(
                        headway_deviation, sched_headway_for_bus
                    )
                base_travel_sec = segment_free_flow_seconds(stop_idx, config)
                seg_congestion_idx = day_params["segment_congestion_index"][stop_idx]
                delay_factor = sample_segment_delay_factor(
                    hour_float, rng, seg_congestion_idx,
                    incident_prob=day_params["incident_prob"],
                    incident_scale=day_params["incident_scale"],
                )
                actual_travel_sec = (base_travel_sec / recovery_mult) * delay_factor
                t += actual_travel_sec / 60.0

    return events


def generate_dataset(n_days, seed, out_path, config=ROUTE_CONFIG, domain_randomization=True):
    rng = np.random.default_rng(seed)
    all_events = []
    burst_days = []
    for day in range(n_days):
        day_events = simulate_one_day(config, day, rng, domain_randomization=domain_randomization)
        if day_events and day_events[0]["is_burst_day"]:
            burst_days.append(day)
        all_events.extend(day_events)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = list(all_events[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_events)

    print(f"Wrote {len(all_events)} events across {n_days} simulated days -> {out_path}")
    if domain_randomization:
        print(f"Domain randomization: ON  |  burst days (festival/exam-style passenger surge): "
              f"{burst_days if burst_days else 'none this run'}")
    else:
        print("Domain randomization: OFF (fixed calibrated config, no per-day variation)")

    deviations = [e["headway_deviation_min"] for e in all_events if e["headway_deviation_min"] != ""]
    sched_headways = [e["headway_scheduled_min"] for e in all_events if e["headway_deviation_min"] != ""]
    bunching_events = [d for d, h in zip(deviations, sched_headways) if abs(d) > h * 0.5]
    print(f"Headway deviation events with data: {len(deviations)}")
    print(f"Events flagged as significant bunching risk (>50% of scheduled headway off): "
          f"{len(bunching_events)} ({100*len(bunching_events)/max(len(deviations),1):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic MTC 21G route dataset")
    parser.add_argument("--days", type=int, default=30, help="Number of service days to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="../data/synthetic_route.csv", help="Output CSV path")
    parser.add_argument("--no-domain-randomization", action="store_true",
                         help="Disable per-day randomization, use the fixed calibrated config only")
    args = parser.parse_args()

    generate_dataset(args.days, args.seed, args.out,
                      domain_randomization=not args.no_domain_randomization)
