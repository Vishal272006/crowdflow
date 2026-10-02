"""
traffic_model.py

Models traffic-induced speed variation on top of the free-flow base speed
from route_model.py. Two components:

1. Smooth time-of-day congestion factor (rush hour slowdowns)
2. Stochastic noise + occasional "incident" delays (signal, blockage, etc.)

This is the piece most worth calibrating carefully, since traffic
variability is the actual root cause of bunching in the simulation -
if this is too smooth/uniform, bunching won't emerge naturally and your
LSTM will have nothing real to learn.
"""

import numpy as np


def time_of_day_factor(hour_float, segment_congestion_index=1.0):
    """
    Returns a multiplier on travel time (>=1.0) based on hour of day AND
    the fixed, real per-segment congestion index (see route_model.py).

    Two things happen together, matching how real chokepoints behave:
    1. A baseline chronic offset -- a bottleneck segment (e.g. Kathipara
       Junction corridor) runs a bit slower than free-flow even off-peak.
    2. Peak amplification scales WITH the segment index -- the same rush
       hour makes a chronic bottleneck much worse, while a quiet stretch
       barely reacts to peak hour at all. This is why bunching often
       starts at the same 2-3 points on a route every day, not randomly.

    Calibrated overall amplitude against real MTC Route 21G figures:
    ~22.4 km/h schedule-average speed vs. ~18 km/h stated peak speed
    (a ~1.24x ratio) for an AVERAGE segment (index ~1.0).
    """
    morning_peak = np.exp(-0.5 * ((hour_float - 8.5) / 1.0) ** 2)
    evening_peak = np.exp(-0.5 * ((hour_float - 18.5) / 1.2) ** 2)
    peak_component = 0.22 * morning_peak + 0.26 * evening_peak

    # Clamp at 0: a below-average-congestion segment (index < 1.0) should
    # get LESS peak amplification, not an actual speed-up below the
    # documented free-flow baseline -- segment_free_flow_min was already
    # extracted from the least-congested real trip, so it IS the floor.
    chronic_baseline = 1.0 + max(0.0, 0.12 * (segment_congestion_index - 1.0))
    peak_scaled = peak_component * segment_congestion_index

    return chronic_baseline + peak_scaled


def sample_segment_delay_factor(hour_float, rng, segment_congestion_index=1.0, incident_prob=0.006, incident_scale=1.7):
    """
    Returns a multiplicative delay factor to apply to free-flow travel time
    for one stop-to-stop segment.

    - Baseline: time_of_day_factor (now segment-aware) * log-normal noise
      (models normal variability: signal timing, minor congestion)
    - Occasionally (incident_prob): a much larger delay factor, modeling a
      blocked lane, breakdown, accident, or signal malfunction. Incident
      probability scales slightly with chronic congestion, since higher-
      traffic segments see more breakdowns/minor accidents in practice.
    """
    base = time_of_day_factor(hour_float, segment_congestion_index)
    noise = rng.lognormal(mean=0.0, sigma=0.07)
    factor = base * noise

    effective_incident_prob = incident_prob * (0.7 + 0.3 * segment_congestion_index)
    if rng.random() < effective_incident_prob:
        low = min(1.2, incident_scale * 0.8)
        factor *= rng.uniform(low, incident_scale)

    return factor


def sample_dwell_noise(rng, sigma=0.10):
    """Small multiplicative noise on dwell time (driver behavior variability)."""
    return rng.lognormal(mean=0.0, sigma=sigma)


def schedule_recovery_speed_multiplier(headway_deviation_min, scheduled_headway_min, gain=0.28, max_adjust=0.35):
    """
    Models a driver's tendency to partially correct for lateness/earliness:
    a late bus (positive deviation) speeds up a bit on the next open segment,
    an early bus eases off. This is a real, well-documented driver behavior
    and also acts as the damping term that keeps bunching severe-but-bounded
    instead of runaway.

    Returns a multiplier applied to base_speed_kmph (e.g. 1.1 = 10% faster).
    """
    if scheduled_headway_min <= 0:
        return 1.0
    normalized_dev = headway_deviation_min / scheduled_headway_min
    adjust = np.clip(gain * normalized_dev, -max_adjust, max_adjust)
    return 1.0 + adjust
