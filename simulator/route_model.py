"""
route_model.py

Defines the physical/operational model of a single bus route:
- stop sequence and real segment travel times
- real dispatch schedule (actual, non-uniform departure times)
- passenger arrival behavior at each stop
- dwell (boarding) time model

CALIBRATION SOURCE (v2, expanded route):
  - The 18-stop version of this route (Broadway <-> Tambaram, 33.6 km) was
    originally calibrated against the published timetable at
    spiritofchennai.com/busroutes/21g-broadway-tambaram/, which gave real
    per-segment travel times (from the 5:45 AM least-congested trip) and
    all 60 real daily dispatch times.
  - This version (v2) EXPANDS the route to all 47 real stops actually
    served by MTC 21G end-to-end, Parry's Corner <-> Kilambakkam Bus
    Terminus, sourced from a live transit app's full stop list (in-chat
    screenshots, Sept 2026) -- a much finer granularity than the
    18-stop timetable, which only names major/interchange stops.

HOW THE 47-STOP TIMINGS WERE DERIVED (read this before trusting the
numbers to more precision than they deserve):
  1. Each of the ORIGINAL 18 timetable stops was matched to its closest
     real stop in the new 47-stop list (many are exact or near-exact
     matches, e.g. old "Adyar Gate" = new "Adyar Gate", old "QMC College"
     = new "Triplicane Queen Marys College").
  2. Between two matched anchor points, the OLD segment's real travel
     time was split EQUALLY across however many new sub-stops fall
     inside it. This is a documented approximation, not measured data --
     we do not have real per-stop timing at this finer granularity, only
     at the original 18-stop resolution.
  3. The route was also EXTENDED beyond the old Tambaram terminus, out to
     the real Kilambakkam Bus Terminus (7 additional real stops: Tambaram
     West Bus Stand -> Irumbuliyur -> Perungalatgur Lake View Stop ->
     Perungalattur -> Perungalathur Iraniyamman Temple -> Vandalur Gate ->
     Vandalur Zoo -> Kilambakkam Bus Terminus). NO real schedule data
     exists for this stretch in our sources -- its ~22 minute total
     travel time and ~8.4 km distance are ESTIMATED from typical GST-Road
     suburban bus speeds (~22-24 km/h), not measured. Flag this
     explicitly if asked in review: this is the least-calibrated part of
     the route model and the best candidate for a real Google Routes API
     / OSRM query if you want to tighten it further.
  4. segment_congestion_index values are inherited from whichever OLD
     segment's chokepoint character each new sub-segment falls inside
     (e.g. all new sub-segments between old "Guindy RS" and old
     "St Thomas Mount" inherit that segment's 1.50 Kathipara Junction
     index). The new Tambaram->Kilambakkam extension has no old-segment
     equivalent; it's assigned 1.20 (moderate-high), reasoned from GST
     Road's general character as a heavy-traffic corridor, but this is a
     documented guess, not a measurement.

IMPORTANT MODELING NOTE (carried over from v1): the original published
schedule uses an identical 90-minute segment-offset template for every
trip regardless of departure time, so it does NOT encode real trip-to-trip
congestion variation -- that variation is applied separately via
traffic_model.py's time-of-day and domain-randomization layers.
"""

import numpy as np
import json
import os

# ---------------------------------------------------------------------------
# OPTIONAL: real Google Routes API calibration data. If you've run
# calibration/geocode_stops.py and calibration/fetch_route_segments.py with
# your own API key, this section upgrades segment_free_flow_min from
# "estimated from anchor interpolation" to "measured from a live traffic
# model" -- automatically, no code changes needed. If those files don't
# exist, the hand-calibrated values below are used unchanged -- nothing
# breaks either way. Given the new extension is the least-calibrated part
# of the route (see module docstring), running the calibration scripts is
# especially worthwhile for those last 7 stops.
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_REAL_SEGMENTS_PATH = os.path.join(_DATA_DIR, "real_segment_travel_times.json")
_REAL_COORDS_PATH = os.path.join(_DATA_DIR, "real_stop_coordinates.json")


def _load_real_segment_minutes(stop_names, fallback_minutes, off_peak_hour="13"):
    """Returns segment_free_flow_min sourced from real_segment_travel_times.json
    if it exists and covers every segment, using the off-peak hour's duration
    as the free-flow baseline (consistent with how the schedule-derived
    fallback was built -- from the LEAST congested reference trip)."""
    if not os.path.exists(_REAL_SEGMENTS_PATH):
        return fallback_minutes, False

    try:
        with open(_REAL_SEGMENTS_PATH) as f:
            real_data = json.load(f)
        minutes = []
        for i in range(len(stop_names) - 1):
            key = f"{stop_names[i]} -> {stop_names[i + 1]}"
            if key not in real_data or off_peak_hour not in real_data[key]:
                return fallback_minutes, False  # incomplete data, don't half-apply it
            minutes.append(real_data[key][off_peak_hour]["duration_sec"] / 60.0)
        return minutes, True
    except (json.JSONDecodeError, KeyError, TypeError):
        return fallback_minutes, False

# ---------------------------------------------------------------------------
# ROUTE CONFIG -- v2, all 47 real stops, Parry's Corner -> Kilambakkam
# ---------------------------------------------------------------------------

STOP_NAMES = [
    "Parrys Corner", "R B I Parrys", "Secretariat", "War Memorial",
    "Madras University", "Marina Beach", "Kannagi Statue or Presidency College",
    "Vivekananda House Bus Terminus", "Triplicane Queen Marys College",
    "DGP Office", "City Centre or Kalyani Hospital", "Royapettah Yellow Pages",
    "Thiruvalluvar Statue", "Luz Corner", "Mylapore Tank", "Mandaveli Market",
    "Adyar Gate", "Nandanam Park", "Kotturpuram",
    "Anna Centenary Library", "B M Birla Planetarium", "CLRI or IIT Madras",
    "Anna University", "Saidapet Court or Saidapet Depot", "Chellammal College",
    "Guindy B.T", "Alandur Metro R.S", "St. Thomas Mount P.O", "Alandur Depot",
    "Meenambakkam International Airport", "Thirusoolam National Airport",
    "Pallavaram", "Chromepet Ponds Company",
    "Saravana Store Chromepet", "Chromepet", "Chromepet MIT Gate",
    "Tambaram TB Hospital", "Tambaram Sanatorium B.T", "Kadaperi",
    "Tambaram West Bus Stand", "Irumbuliyur", "Perungalatgur Lake View Stop",
    "Perungalattur", "Perungalathur Iraniyamman Temple", "Vandalur Gate",
    "Vandalur Zoo", "Kilambakkam Bus Terminus"
]

# 46 segment free-flow times (minutes), derived per the anchor-interpolation
# method described in the module docstring. The final 7 values (Tambaram
# West Bus Stand -> ... -> Kilambakkam Bus Terminus) are an ESTIMATED
# extension with no real schedule backing -- see docstring point 3.
_SEGMENT_FREE_FLOW_FALLBACK = [
    2.5, 2.5,                          # Parrys Corner..Secretariat (real anchor: 5 min)
    1.3, 1.3, 1.4,                     # Secretariat..Marina Beach (real anchor: 4 min)
    1.3, 1.3, 1.4,                     # Marina Beach..Triplicane QMC (real anchor: 4 min)
    1.3, 1.3, 1.4,                     # Triplicane QMC..Royapettah Yellow Pages (real anchor: 4 min)
    1.25, 1.25, 1.25, 1.25,            # Royapettah..Mandaveli Market (real anchor: 5 min)
    6.0,                                # Mandaveli Market..Adyar Gate (real anchor: 6 min)
    2.5, 2.5,                          # Adyar Gate..Kotturpuram (real anchor: 5 min)
    2.0, 1.0, 1.0, 1.0,               # Kotturpuram..Anna University (merged same-stop alias)
    5.0,                                # Anna University..Saidapet Court (real anchor: 5 min)
    2.5, 2.5,                          # Saidapet Court..Guindy B.T (real anchor: 5 min)
    3.5, 3.5,                          # Guindy B.T..St Thomas Mount P.O (real anchor: 7 min)
    3.0, 3.0,                          # St Thomas Mount P.O..Meenambakkam Airport (real anchor: 6 min)
    5.0,                                # Meenambakkam..Thirusoolam Airport (real anchor: 5 min)
    7.0,                                # Thirusoolam..Pallavaram (combined same-stop alias)
    2.0, 2.0, 2.0,                     # Pallavaram..Chromepet (real anchor: 6 min)
    1.7, 1.7, 1.6,                     # Chromepet..Tambaram Sanatorium B.T (real anchor: 5 min)
    3.0, 3.0,                          # Tambaram Sanatorium..Tambaram West Bus Stand (real anchor: 6 min)
    # --- ESTIMATED extension, no real schedule data (see docstring) ---
    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0,  # Tambaram West Bus Stand..Kilambakkam Bus Terminus (est. 22 min)
]

ROUTE_CONFIG = {
    "route_id": "MTC_21G_ParrysCorner_Kilambakkam",
    "stop_names": STOP_NAMES,
    # ESTIMATED: old 33.6 km / 90 min anchor implies ~0.373 km/min average;
    # applied to the new ~112 min total gives ~41.8 km. Not a measured
    # distance -- verify with calibration/fetch_route_segments.py if you
    # want a real figure for your report.
    "total_distance_km": 41.8,
    "segment_free_flow_min": _load_real_segment_minutes(
        STOP_NAMES, _SEGMENT_FREE_FLOW_FALLBACK,
    )[0],
    # real dispatch times, Parrys Corner -> Kilambakkam direction, as minutes
    # since midnight. UNCHANGED from the original 18-stop calibration --
    # dispatch frequency is a property of the SERVICE, not stop granularity,
    # so the original 60 real daily departure times still apply.
    "real_dispatch_times_min": [
        345, 360, 375, 390, 405, 435, 460, 490, 510, 520, 527, 535, 545, 555,
        560, 570, 580, 610, 630, 660, 670, 685, 695, 700, 710, 720, 725, 735,
        765, 800, 825, 845, 865, 875, 885, 895, 910, 930, 945, 965, 980, 1005,
        1020, 1025, 1037, 1045, 1060, 1075, 1090, 1110, 1135, 1150, 1185, 1200,
        1205, 1206, 1212, 1230, 1242, 1285
    ],
    # PER-SEGMENT chronic congestion index (46 values). Inherited from
    # whichever OLD (18-stop) segment's real-world chokepoint character
    # each new sub-segment falls inside -- see module docstring point 4.
    # The final 7 values (new extension) have no old-segment equivalent;
    # 1.20 is a documented estimate for a GST-Road suburban stretch, not
    # a measurement.
    "segment_congestion_index": [
        1.05, 1.05,                                    # -> Secretariat (city core, govt)
        1.00, 1.00, 1.00,                              # -> Marina Beach
        0.90, 0.90, 0.90,                              # -> Triplicane QMC (quiet, Marina-side)
        0.85, 0.85, 0.85,                              # -> Royapettah Yellow Pages
        1.00, 1.00, 1.00, 1.00,                        # -> Mandaveli Market
        1.30,                                           # -> Adyar Gate (Adyar signal, commercial)
        1.05, 1.05,                                    # -> Kotturpuram
        1.00, 1.00, 1.00, 1.00,                        # -> Anna University
        1.10,                                           # -> Saidapet Court (IT corridor)
        1.15, 1.15,                                    # -> Guindy B.T (industrial, approach to Kathipara)
        1.50, 1.50,                                    # -> St Thomas Mount P.O (Kathipara Junction -- worst)
        1.35, 1.35,                                    # -> Meenambakkam Airport (GST Rd, Kathipara-adjacent)
        1.05,                                           # -> Thirusoolam Airport
        1.15,                                           # -> Pallavaram (GST Road)
        1.40, 1.40, 1.40,                              # -> Chromepet (GST Road heavy-truck corridor)
        1.10, 1.10, 1.10,                              # -> Tambaram Sanatorium B.T
        1.00, 1.00,                                    # -> Tambaram West Bus Stand
        1.20, 1.20, 1.20, 1.20, 1.20, 1.20, 1.20,      # -> Kilambakkam (ESTIMATED extension)
    ],
    "n_buses_concurrent": 12,       # rough estimate of buses in service at once
    "service_start_hour": 5.75,     # 5:45 AM
    "service_end_hour": 22.917,     # last dispatch ~9:25 PM + longer trip time
    # peak-hour average speed (km/h), per original 18-stop schedule notes
    "peak_avg_speed_kmph": 18.0,
    # implied full-schedule average speed, recomputed for the extended
    # ~112 min / ~41.8 km route (was 33.6/1.5 for the old 18-stop version)
    "schedule_avg_speed_kmph": 41.8 / (112.0 / 60.0),
    # passenger arrival rate at each stop, per minute (Poisson lambda).
    # NOT from any published source -- estimated by stop role (interchange/
    # institutional/residential/terminus), same approach as v1. 47 values,
    # one per stop, SCALED so total system-wide boarding demand matches the
    # original 18-stop calibration (sum ~36.0) rather than being
    # accidentally multiplied when split across more, finer stops -- an
    # early draft of this list summed to ~82.8 (2.3x too much), which
    # caused runaway bunching until caught and fixed. Flag the role-based
    # shape as an assumption in your report; the ABSOLUTE scale is anchored
    # to the original calibration.
    "passenger_arrival_rate": [
        1.52, 0.87, 0.43, 0.35, 0.65,   # Parrys Corner..Madras University
        0.87, 0.65, 0.52, 0.57, 0.35,   # Marina Beach..DGP Office
        0.65, 0.65, 0.43, 0.87, 0.65,   # City Centre..Mylapore Tank
        1.30, 0.96, 0.43, 0.52, 0.78,   # Mandaveli Market..Anna Centenary Library (merged Kottur alias)
        0.35, 0.65, 1.52, 0.87, 0.65,   # B M Birla Planetarium..Chellammal College
        0.65, 1.30, 1.52, 0.78, 0.43,   # Chellammal College..Alandur Depot
        0.65, 0.87, 1.74, 0.65,         # Meenambakkam..Chromepet Ponds Company; merged Pallavaram aliases
        0.87, 1.22, 1.09, 0.52, 0.52,   # Saravana Store..Tambaram Sanatorium B.T
        0.43, 1.30, 0.43, 0.35, 0.52,   # Kadaperi..Perungalattur
        0.43, 0.52, 0.65,               # Vandalur Gate..Kilambakkam Bus Terminus
    ],
    "boarding_time_per_pax_sec": 3.0,
    "fixed_dwell_overhead_sec": 8.0,
    "bus_capacity": 60,
}


def n_stops(config=ROUTE_CONFIG):
    return len(config["stop_names"])


def segment_free_flow_seconds(stop_idx, config=ROUTE_CONFIG):
    """Real free-flow travel time (seconds) from stop_idx to stop_idx+1."""
    return config["segment_free_flow_min"][stop_idx] * 60.0


def sample_boarding_passengers(stop_idx, elapsed_minutes_since_last_bus, rng, config=ROUTE_CONFIG):
    """
    Passengers waiting at a stop depend on how long it's been since the
    previous bus served that stop (classic bunching feedback mechanism:
    a late bus finds MORE waiting passengers -> longer dwell -> later still).
    """
    rate_per_min = config["passenger_arrival_rate"][stop_idx]
    expected = rate_per_min * max(elapsed_minutes_since_last_bus, 0.0)
    return rng.poisson(max(expected, 0.01))


def dwell_time_seconds(n_boarding, config=ROUTE_CONFIG):
    return config["fixed_dwell_overhead_sec"] + n_boarding * config["boarding_time_per_pax_sec"]
