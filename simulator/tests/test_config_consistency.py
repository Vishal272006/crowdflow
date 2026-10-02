"""
test_config_consistency.py

Cheap insurance against silent config drift. These tests don't check
whether the SIMULATED VALUES are "correct" (that's a calibration
question, not a unit-test question) -- they check that the CONFIG
ARRAYS in route_model.py are internally consistent with each other,
which is the kind of bug that's invisible until generate_dataset.py
crashes with an IndexError three files away from where the actual
mistake was made (e.g. someone adds a stop to stop_names but forgets
to add a matching entry to passenger_arrival_rate).

Run with:
    python -m pytest test_config_consistency.py -v
or, without pytest installed:
    python test_config_consistency.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from route_model import ROUTE_CONFIG, n_stops
from traffic_model import time_of_day_factor, schedule_recovery_speed_multiplier


def test_stop_count_consistent():
    n = n_stops(ROUTE_CONFIG)
    assert n == len(ROUTE_CONFIG["stop_names"]), \
        f"n_stops()={n} but stop_names has {len(ROUTE_CONFIG['stop_names'])} entries"


def test_passenger_arrival_rate_matches_stop_count():
    n = n_stops(ROUTE_CONFIG)
    assert len(ROUTE_CONFIG["passenger_arrival_rate"]) == n, \
        (f"passenger_arrival_rate has {len(ROUTE_CONFIG['passenger_arrival_rate'])} "
         f"entries, expected {n} (one per stop)")


def test_segment_arrays_match_n_segments():
    """Segment-level arrays (travel time, congestion) should have exactly
    n_stops - 1 entries -- one per gap BETWEEN stops, not one per stop."""
    n_segments = n_stops(ROUTE_CONFIG) - 1

    assert len(ROUTE_CONFIG["segment_free_flow_min"]) == n_segments, \
        (f"segment_free_flow_min has {len(ROUTE_CONFIG['segment_free_flow_min'])} "
         f"entries, expected {n_segments} segments")

    assert len(ROUTE_CONFIG["segment_congestion_index"]) == n_segments, \
        (f"segment_congestion_index has {len(ROUTE_CONFIG['segment_congestion_index'])} "
         f"entries, expected {n_segments} segments")


def test_segment_free_flow_times_sum_to_documented_total():
    """v2 (47-stop route, Parry's Corner -> Kilambakkam): accept either the
    ~112 min fallback timetable or complete OSRM routed durations (currently
    ~97 min). Both should remain within a plausible end-to-end range."""
    total = sum(ROUTE_CONFIG["segment_free_flow_min"])
    assert 90 <= total <= 120, \
        f"segment_free_flow_min sums to {total} min, expected 90-120 min (v2 route)"


def test_congestion_index_values_are_positive():
    for i, v in enumerate(ROUTE_CONFIG["segment_congestion_index"]):
        assert v > 0, f"segment_congestion_index[{i}] = {v}, must be positive"


def test_dispatch_times_are_sorted_and_within_service_hours():
    times = ROUTE_CONFIG["real_dispatch_times_min"]
    assert times == sorted(times), "real_dispatch_times_min must be in chronological order"
    assert len(times) > 0, "real_dispatch_times_min must not be empty"
    assert min(times) >= 0 and max(times) <= 1440, \
        "dispatch times must be valid minutes-since-midnight (0-1440)"


def test_time_of_day_factor_never_below_one():
    """Congestion multiplier should never REDUCE travel time below free-flow."""
    for hour in [0, 6, 8.5, 12, 18.5, 23]:
        for idx in [0.85, 1.0, 1.5]:
            factor = time_of_day_factor(hour, idx)
            assert factor >= 1.0, f"time_of_day_factor({hour}, {idx}) = {factor}, expected >= 1.0"


def test_schedule_recovery_bounded():
    """Recovery speed-up should never exceed the documented max_adjust cap,
    regardless of how extreme the input deviation is."""
    extreme_mult = schedule_recovery_speed_multiplier(
        headway_deviation_min=-1000, scheduled_headway_min=8
    )
    assert 0.5 <= extreme_mult <= 1.5, \
        f"schedule_recovery_speed_multiplier returned {extreme_mult} for an extreme input -- bound not holding"


def _run_all():
    """Fallback runner if pytest isn't installed."""
    import traceback
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -- {e}")
            failed += 1
        except Exception:
            print(f"ERROR: {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
