"""
fetch_osrm_segments.py

Gets REAL road-network distance and free-flow driving time for each of the
48 stop-to-stop segments of MTC Route 21G, using OSRM (Open Source Routing
Machine) -- free, no API key, no billing account. Uses the real GPS
coordinates from geocode_stops.py and actual road routing (real roads,
real turns, real distance) instead of our schedule-interpolated or
hand-estimated segment times.

IMPORTANT, READ BEFORE RUNNING -- what this DOES and DOES NOT do:
  - DOES give real road-network distance and a free-flow (no-traffic)
    driving time estimate, based on road type and speed limit.
  - Does NOT give time-of-day traffic variation. OSRM has no live or
    historical traffic feed -- it returns the SAME number for 8 AM and
    10 PM. The "1 PM Kathipara vs 10 PM Kathipara" difference your
    project needs still comes entirely from traffic_model.py's
    time-of-day curve and congestion index, exactly as before. This
    script only upgrades the FOUNDATION those layers operate on (real
    base segment times) from estimated to measured.
  - A bus is slower than a free-flow car (stops, lower speed limits in
    practice, etc.) -- OSRM's numbers will likely run FASTER than our
    real schedule-derived times for the same stretch. This script prints
    a comparison against the current fallback so you can see the gap and
    judge whether a correction factor is warranted (see the printed
    summary at the end).

REQUIRES: real_stop_coordinates.json from geocode_stops.py -- run that
FIRST. Makes network calls to the public OSRM demo server
(router.project-osrm.org) -- will NOT run inside an offline/sandboxed
environment, and is a shared free service, so this script rate-limits
itself to be polite (not just to avoid being blocked).

Usage:
    python fetch_osrm_segments.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
from route_model import ROUTE_CONFIG  # noqa: E402

OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving"
COORDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "real_stop_coordinates.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "real_segment_travel_times.json")
RATE_LIMIT_SECONDS = 1.0

# route_model.py's _load_real_segment_minutes() always looks under this
# exact key (its off_peak_hour default) -- OSRM gives one free-flow value
# with no time-of-day variation, so we just store it under that one key.
# This means route_model.py needs ZERO code changes to pick this up.
FIXED_HOUR_KEY = "13"


def fetch_one_segment(origin, destination):
    """origin/destination: {"lat":..., "lng":...}. Returns
    {"duration_sec": float, "distance_m": float} or None on failure."""
    coord_str = f"{origin['lng']},{origin['lat']};{destination['lng']},{destination['lat']}"
    url = f"{OSRM_BASE_URL}/{coord_str}?overview=false"

    req = urllib.request.Request(url, headers={
        "User-Agent": "crowdflow-capstone-project (student project, MTC 21G bunching prediction)"
    })
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode())

    if data.get("code") != "Ok" or not data.get("routes"):
        return None

    route = data["routes"][0]
    return {"duration_sec": float(route["duration"]), "distance_m": float(route["distance"])}


def main():
    if not os.path.exists(COORDS_PATH):
        print(f"ERROR: {COORDS_PATH} not found. Run geocode_stops.py FIRST.")
        sys.exit(1)

    with open(COORDS_PATH) as f:
        coords = json.load(f)

    stop_names = ROUTE_CONFIG["stop_names"]
    missing = [s for s in stop_names if s not in coords]
    if missing:
        print(f"ERROR: missing coordinates for: {missing}")
        print(f"Re-run geocode_stops.py and check it succeeded for all {len(stop_names)} stops.")
        sys.exit(1)

    # resume from a previous partial run
    results = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)

    n_segments = len(stop_names) - 1
    old_fallback = ROUTE_CONFIG["segment_free_flow_min"]  # current estimate, for comparison
    new_minutes = []
    failures = []

    print(f"Fetching {n_segments} segments from OSRM (public server, rate-limited "
          f"to {RATE_LIMIT_SECONDS}s/request -- this will take a minute or two)...")

    for i in range(n_segments):
        origin_name, dest_name = stop_names[i], stop_names[i + 1]
        segment_key = f"{origin_name} -> {dest_name}"

        if segment_key in results and FIXED_HOUR_KEY in results[segment_key]:
            minutes = results[segment_key][FIXED_HOUR_KEY]["duration_sec"] / 60.0
            new_minutes.append(minutes)
            print(f"[{i+1}/{n_segments}] {segment_key} -- already fetched, skipping")
            continue

        print(f"[{i+1}/{n_segments}] {segment_key}...", end="  ", flush=True)
        try:
            r = fetch_one_segment(coords[origin_name], coords[dest_name])
        except Exception as e:
            print(f"ERROR: {e}")
            r = None

        if r is None:
            print("FAILED -- no route found")
            failures.append(segment_key)
            new_minutes.append(old_fallback[i])  # keep old estimate for this one segment
        else:
            minutes = r["duration_sec"] / 60.0
            old_minutes = old_fallback[i]
            diff_pct = 100 * (minutes - old_minutes) / old_minutes if old_minutes else 0
            print(f"{minutes:.2f} min, {r['distance_m']/1000:.2f} km  "
                  f"(old estimate: {old_minutes:.2f} min, {diff_pct:+.0f}%)")
            results[segment_key] = {FIXED_HOUR_KEY: r}
            new_minutes.append(minutes)

        time.sleep(RATE_LIMIT_SECONDS)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved -> {OUT_PATH}")
    if failures:
        print(f"\n{len(failures)} segment(s) failed and fell back to the old estimate: {failures}")
        print("Re-run this script -- successful segments won't be re-queried, only these will retry.")

    old_total = sum(old_fallback)
    new_total = sum(new_minutes)
    print(f"\n{'=' * 60}")
    print(f"COMPARISON: old estimated total = {old_total:.1f} min   "
          f"OSRM real total = {new_total:.1f} min   "
          f"({100*(new_total-old_total)/old_total:+.1f}%)")
    print(f"{'=' * 60}")
    if new_total < old_total * 0.8:
        print("NOTE: OSRM's free-flow car routing came back notably FASTER than the real")
        print("published bus schedule -- expected, since a bus makes stops and typically")
        print("drives slower than free-flow car traffic. Consider applying a bus-vs-car")
        print("correction factor (e.g. multiply OSRM times by ~1.2-1.4x) if you want the")
        print("baseline to better reflect real bus travel rather than car travel -- discuss")
        print("this tradeoff explicitly in your report rather than silently picking one.")


if __name__ == "__main__":
    main()
