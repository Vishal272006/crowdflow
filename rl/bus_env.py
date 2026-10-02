"""A day-level, departure-only simulator for the CrowdFlow RL controller.

One episode is one full service day. Before every bus after the first leaves
Parrys Corner, the agent chooses a departure delay. Once that bus departs it
is never intentionally held again. Each choice changes passenger buildup,
dwell time, and the headway experienced by later buses that day.
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
from route_model import ROUTE_CONFIG, n_stops, segment_free_flow_seconds, sample_boarding_passengers, dwell_time_seconds  # noqa: E402
from traffic_model import sample_segment_delay_factor, sample_dwell_noise, schedule_recovery_speed_multiplier  # noqa: E402
from generate_dataset import sample_daily_randomization  # noqa: E402


DEPARTURE_DELAY_SEC = [0, 30, 60, 90, 120]
# The explicit planner searches a continuous practical range. It first checks
# every five seconds, then checks each individual second around the best
# result. This gives a dynamic recommendation such as 143 seconds instead of
# forcing the dispatcher into a small fixed list of choices.
PLANNER_MAX_DELAY_SEC = 15 * 60
# A new bus must remain at least this fraction of its scheduled gap behind the
# prior bus at every stop.  The planner can be made stricter or looser later
# when real MTC operating data is available.
SAFE_MIN_GAP_RATIO = 0.50
N_STOPS = n_stops(ROUTE_CONFIG)
BUS_CAPACITY = ROUTE_CONFIG["bus_capacity"]
MAX_HEADWAY_MIN = 45.0


def load_trips(csv_path):
    """Load CSV trips for the dashboard's historical simulated-trip display."""
    df = pd.read_csv(csv_path)
    arrivals = {
        (int(row.day), int(row.bus_id), int(row.stop_idx)): float(row.arrival_time_min)
        for row in df.itertuples(index=False)
    }
    scheduled = (df.drop_duplicates(["day", "bus_id"])
                 .set_index(["day", "bus_id"])["headway_scheduled_min"].to_dict())
    max_bus = df.groupby("day")["bus_id"].max().to_dict()
    trips = []
    for (day, bus_id), group in df[df["bus_id"] != 1].groupby(["day", "bus_id"]):
        group = group.sort_values("stop_idx").reset_index(drop=True)
        if len(group) != N_STOPS:
            continue
        if bus_id < max_bus[day]:
            following_schedule = scheduled[(day, bus_id + 1)]
            group = group.copy()
            group["gap_to_behind_min"] = [
                arrivals[(day, bus_id + 1, stop_idx)] - arrivals[(day, bus_id, stop_idx)] - following_schedule
                for stop_idx in range(N_STOPS)
            ]
        else:
            group = group.copy()
            group["gap_to_behind_min"] = 0.0
        trips.append(group)
    return trips


def load_dispatch_cases(csv_path):
    """Build dashboard-ready departure states from recorded simulated trips."""
    df = pd.read_csv(csv_path)
    trip_lookup = {
        (int(day), int(bus_id)): group.sort_values("stop_idx").reset_index(drop=True)
        for (day, bus_id), group in df.groupby(["day", "bus_id"])
    }
    max_bus = df.groupby("day")["bus_id"].max().to_dict()
    cases = []
    for (day, bus_id), trip in trip_lookup.items():
        if bus_id == 1 or len(trip) != N_STOPS:
            continue
        previous = trip_lookup[(day, bus_id - 1)]
        row = trip.iloc[0]
        gap = float(row["headway_scheduled_min"])
        planned_departure = float(row["scheduled_dispatch_min"])
        previous_origin = float(previous.iloc[0]["arrival_time_min"])
        ahead_deviation = planned_departure - previous_origin - gap
        previous_progress = float(np.mean(previous["arrival_time_min"] <= planned_departure))
        following_gap = (float(trip_lookup[(day, bus_id + 1)].iloc[0]["headway_scheduled_min"])
                         if bus_id < max_bus[day] else gap)
        estimated_boarding = ROUTE_CONFIG["passenger_arrival_rate"][0] * max(planned_departure - previous_origin, 0.0)
        cases.append({
            "key": (day, bus_id),
            "previous_progress": previous_progress,
            "observation": np.array([
                ahead_deviation / max(gap, 1e-6),
                previous_progress,
                min(estimated_boarding, BUS_CAPACITY) / BUS_CAPACITY,
                (planned_departure % 1440) / 1440.0,
                gap / MAX_HEADWAY_MIN,
                following_gap / MAX_HEADWAY_MIN,
            ], dtype=np.float32),
        })
    return cases


class BusDispatchEnv(gym.Env):
    """Choose one pre-departure delay for each bus in a simulated service day."""

    metadata = {"render_modes": []}

    def __init__(self, seed=None, delay_cost_weight=0.15, behind_weight=0.55):
        super().__init__()
        self.delay_cost_weight = delay_cost_weight
        self.behind_weight = behind_weight
        self.action_space = spaces.Discrete(len(DEPARTURE_DELAY_SEC))
        # prior-bus spacing, prior-bus route progress, expected terminus demand,
        # time of day, planned gap before, planned gap after
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(6,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.dispatch_times = ROUTE_CONFIG["real_dispatch_times_min"]
        self.last_bus_arrival = None
        self.previous_trip = None
        self.bus_ptr = None
        self.day_params = None

    def _scheduled_gap(self, bus_ptr):
        return self.dispatch_times[bus_ptr] - self.dispatch_times[bus_ptr - 1]

    def _simulate_bus(self, bus_ptr, departure_delay_seconds, last_bus_arrival=None, rng=None):
        """Project one full bus trip from the supplied route-state snapshot."""
        bus_id = bus_ptr + 1
        rng = self._rng if rng is None else rng
        last_bus_arrival = self.last_bus_arrival if last_bus_arrival is None else last_bus_arrival
        scheduled_departure = self.dispatch_times[bus_ptr]
        scheduled_gap = self._scheduled_gap(bus_ptr) if bus_ptr else self.dispatch_times[1] - self.dispatch_times[0]
        t = float(scheduled_departure + departure_delay_seconds / 60.0)
        arrivals, deviations = [], []

        for stop_idx in range(N_STOPS):
            # Headways are arrival-to-arrival.  Dwell affects departure for
            # the next segment, rather than redefining when this stop was
            # reached.
            arrival_time = t
            hour_float = (arrival_time % 1440) / 60.0
            elapsed_since_last = (scheduled_gap if last_bus_arrival[stop_idx] is None
                                  else arrival_time - last_bus_arrival[stop_idx])
            n_boarding = sample_boarding_passengers(stop_idx, elapsed_since_last, rng, ROUTE_CONFIG)
            n_boarding = min(int(round(n_boarding * self.day_params["burst_multiplier"])), BUS_CAPACITY)
            dwell_seconds = dwell_time_seconds(n_boarding, ROUTE_CONFIG) * sample_dwell_noise(rng)

            if last_bus_arrival[stop_idx] is None:
                deviation = 0.0
            else:
                deviation = arrival_time - last_bus_arrival[stop_idx] - scheduled_gap
            last_bus_arrival[stop_idx] = arrival_time
            arrivals.append(arrival_time)
            deviations.append(deviation)

            t = arrival_time + dwell_seconds / 60.0

            if stop_idx < N_STOPS - 1:
                recovery = schedule_recovery_speed_multiplier(deviation, scheduled_gap)
                delay_factor = sample_segment_delay_factor(
                    hour_float, rng, self.day_params["segment_congestion_index"][stop_idx],
                    incident_prob=self.day_params["incident_prob"],
                    incident_scale=self.day_params["incident_scale"],
                )
                t += (segment_free_flow_seconds(stop_idx, ROUTE_CONFIG) / recovery) * delay_factor / 60.0

        return {
            "bus_id": bus_id,
            "scheduled_gap": scheduled_gap,
            "arrivals": np.array(arrivals, dtype=np.float32),
            "deviations": np.array(deviations, dtype=np.float32),
        }

    def plan_current_departure(self):
        """Compare every allowed start time before the current bus departs.

        The prior bus's complete projected arrival path is the future-path
        forecast for Bus A. Each candidate creates a fresh projection for
        Bus B from the identical traffic, passenger, and random-event state.
        Therefore differences between candidates come only from their chosen
        start time and the passenger/dwell effects that choice creates.
        """
        if self.bus_ptr is None or self.previous_trip is None:
            raise RuntimeError("Call reset() before planning a departure")

        state_snapshot = list(self.last_bus_arrival)
        rng_state = copy.deepcopy(self._rng.bit_generator.state)

        def selection_key(candidate):
            """Put route safety before punctuality and passenger delay."""
            return (
                candidate["max_unsafe_ratio"],
                candidate["mean_unsafe_ratio"],
                candidate["score"],
            )

        def project(delay_seconds):
            candidate_rng = np.random.default_rng()
            candidate_rng.bit_generator.state = copy.deepcopy(rng_state)
            projected_trip = self._simulate_bus(
                self.bus_ptr,
                delay_seconds,
                last_bus_arrival=list(state_snapshot),
                rng=candidate_rng,
            )
            actual_gap = projected_trip["arrivals"] - self.previous_trip["arrivals"]
            gap_deviation = actual_gap - projected_trip["scheduled_gap"]
            gap_ratio = np.abs(gap_deviation) / projected_trip["scheduled_gap"]
            safe_gap_min = SAFE_MIN_GAP_RATIO * projected_trip["scheduled_gap"]
            unsafe_shortfall = np.maximum(0.0, safe_gap_min - actual_gap)
            closest_stop_idx = int(np.argmin(actual_gap))
            return {
                "delay_seconds": delay_seconds,
                "projected_trip": projected_trip,
                "mean_gap_ratio": float(np.mean(gap_ratio)),
                "max_gap_ratio": float(np.max(gap_ratio)),
                "closest_stop_idx": closest_stop_idx,
                "closest_stop_name": ROUTE_CONFIG["stop_names"][closest_stop_idx],
                "closest_gap_min": float(actual_gap[closest_stop_idx]),
                "scheduled_gap_min": float(projected_trip["scheduled_gap"]),
                "minimum_safe_gap_min": float(safe_gap_min),
                "max_unsafe_ratio": float(np.max(unsafe_shortfall) / projected_trip["scheduled_gap"]),
                "mean_unsafe_ratio": float(np.mean(unsafe_shortfall) / projected_trip["scheduled_gap"]),
                "is_safe": bool(np.all(unsafe_shortfall <= 1e-6)),
                "score": float(np.mean(gap_ratio) + self.delay_cost_weight * (delay_seconds / 60.0)),
            }

        # The range itself is dynamic: start with five minutes, then extend
        # beyond that only when the bus is already too close at departure.
        scheduled_gap = self._scheduled_gap(self.bus_ptr)
        planned_departure = self.dispatch_times[self.bus_ptr]
        origin_gap = planned_departure - float(self.previous_trip["arrivals"][0])
        origin_shortfall_seconds = max(0.0, scheduled_gap - origin_gap) * 60.0
        dynamic_max_delay = min(
            PLANNER_MAX_DELAY_SEC,
            max(300, int(np.ceil(origin_shortfall_seconds + 180))),
        )

        # Broad search across the calculated range, then one-second refinement.
        coarse_candidates = [project(delay) for delay in range(0, dynamic_max_delay + 1, 5)]

        # If the initial search cannot keep a safe gap, continue all the way
        # to the practical limit.  A 1-second recommendation must never win
        # while a projected bus catch-up remains somewhere on the route.
        if (min(candidate["max_unsafe_ratio"] for candidate in coarse_candidates) > 1e-6
                and dynamic_max_delay < PLANNER_MAX_DELAY_SEC):
            coarse_candidates.extend(
                project(delay)
                for delay in range(dynamic_max_delay + 5, PLANNER_MAX_DELAY_SEC + 1, 5)
            )
            dynamic_max_delay = PLANNER_MAX_DELAY_SEC

        coarse_best = min(coarse_candidates, key=selection_key)
        refine_start = max(0, coarse_best["delay_seconds"] - 5)
        refine_end = min(dynamic_max_delay, coarse_best["delay_seconds"] + 5)
        refined_candidates = [project(delay) for delay in range(refine_start, refine_end + 1)]
        candidates = {candidate["delay_seconds"]: candidate for candidate in coarse_candidates}
        candidates.update({candidate["delay_seconds"]: candidate for candidate in refined_candidates})
        candidates = [candidates[delay] for delay in sorted(candidates)]

        # This is the first and most important decision: when leaving at the
        # published time is already safe, tell the driver to depart now.
        # Do not create a meaningless few-second wait merely because it
        # slightly improves an average mathematical score.
        depart_now = candidates[0]
        best = depart_now if depart_now["is_safe"] else min(candidates, key=selection_key)
        warning = None
        if not best["is_safe"]:
            warning = (
                "No safe departure time was found within the 15-minute planning limit. "
                "The displayed delay is the safest available option."
            )
        return {"recommended_departure_delay_seconds": best["delay_seconds"],
                "best_candidate": best, "candidates": candidates, "warning": warning}

    def _observation(self):
        scheduled_gap = self._scheduled_gap(self.bus_ptr)
        planned_departure = self.dispatch_times[self.bus_ptr]
        previous_origin = float(self.previous_trip["arrivals"][0])
        ahead_deviation = planned_departure - previous_origin - scheduled_gap
        previous_progress = float(np.mean(self.previous_trip["arrivals"] <= planned_departure))
        estimated_boarding = ROUTE_CONFIG["passenger_arrival_rate"][0] * max(planned_departure - previous_origin, 0.0)
        following_gap = (self.dispatch_times[self.bus_ptr + 1] - planned_departure
                         if self.bus_ptr < len(self.dispatch_times) - 1 else scheduled_gap)
        return np.array([
            ahead_deviation / max(scheduled_gap, 1e-6),
            previous_progress,
            min(estimated_boarding, BUS_CAPACITY) / BUS_CAPACITY,
            (planned_departure % 1440) / 1440.0,
            scheduled_gap / MAX_HEADWAY_MIN,
            following_gap / MAX_HEADWAY_MIN,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day_params = sample_daily_randomization(ROUTE_CONFIG, self._rng)
        self.last_bus_arrival = [None] * N_STOPS
        self.bus_ptr = 0
        self.previous_trip = self._simulate_bus(bus_ptr=0, departure_delay_seconds=0)
        self.bus_ptr = 1
        return self._observation(), {}

    def _apply_departure_delay(self, delay_seconds):
        current_trip = self._simulate_bus(self.bus_ptr, delay_seconds)
        ahead_ratio = np.abs(current_trip["deviations"]) / current_trip["scheduled_gap"]
        mean_ahead = float(np.mean(ahead_ratio))
        reward = -mean_ahead - self.delay_cost_weight * (delay_seconds / 60.0)

        self.previous_trip = current_trip
        self.bus_ptr += 1
        terminated = self.bus_ptr >= len(self.dispatch_times)
        observation = np.zeros(6, dtype=np.float32) if terminated else self._observation()
        info = {
            "mean_ahead_ratio": mean_ahead,
            "max_ahead_ratio": float(np.max(ahead_ratio)),
            "departure_delay_seconds": delay_seconds,
        }
        return observation, reward, terminated, False, info

    def step(self, action):
        """Gymnasium interface for the discrete RL controller."""
        return self._apply_departure_delay(DEPARTURE_DELAY_SEC[int(action)])

    def depart_with_delay(self, delay_seconds):
        """Planner interface for an exact calculated delay in seconds."""
        return self._apply_departure_delay(int(delay_seconds))
