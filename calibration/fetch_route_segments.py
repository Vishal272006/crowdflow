"""
fetch_route_segments.py

Uses Google's Routes API to get REAL, traffic-aware travel time and
distance estimates for each of the 17 stop-to-stop segments of MTC Route
21G, at several different times of day. This is what upgrades
segment_free_flow_min and the congestion calibration in route_model.py
from "estimated from one schedule trip + published congestion reports"
to "measured from Google's live traffic model."

REQUIRES: real_stop_coordinates.json from geocode_stops.py -- run that
script FIRST.

Requires: a Google Cloud API key with the Routes API enabled. Makes
network calls -- will NOT run inside an offline/sandboxed environment.

SECURITY: reads the key from an environment variable, same as
geocode_stops.py -- see that file's docstring for how to set it.

Usage:
    python fetch_route_segments.py

Output: ../data/real_segment_travel_times.json -- one entry per segment,
with duration/distance at each queried time-of-day. Read automatically
by route_model.py if present.

NOTE ON TIMING: the Routes API's traffic-aware mode uses live traffic for
"now" and PREDICTED traffic for a future departureTime. To get a
representative "typical Tuesday" estimate rather than whatever is
happening right this second, this script queries a departureTime set to
the NEXT occurrence of each target hour on a weekday -- not literally
"right now."
"""

import json
import os
import sys
import time
import datetime
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
from route_model import ROUTE_CONFIG  # noqa: E402

STOP_NAMES = ROUTE_CONFIG["stop_names"]

# times of day to query -- covers off-peak, morning peak, midday, evening peak
QUERY_HOURS = [6, 9, 13, 18, 22]

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
COORDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "real_stop_coordinates.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "real_segment_travel_times.json")


def next_weekday_at_hour(hour):
    """Returns an RFC3339 timestamp for the next upcoming Tuesday at the
    given hour -- a stand-in for 'a typical weekday', avoiding weekends
    and avoiding querying a time that's already in the past today."""
    now = datetime.datetime.utcnow()
    days_ahead = (1 - now.weekday()) % 7  # 1 = Tuesday
    if days_ahead == 0:
        days_ahead = 7
    target_date = now + datetime.timedelta(days=days_ahead)
    target = target_date.replace(hour=hour, minute=0, second=0, microsecond=0)
    return target.strftime("%Y-%m-%dT%H:%M:%S") + "Z"


def fetch_one_segment(origin, destination, hour, api_key):
    body = {
        "origin": {"location": {"latLng": {"latitude": origin["lat"], "longitude": origin["lng"]}}},
        "destination": {"location": {"latLng": {"latitude": destination["lat"], "longitude": destination["lng"]}}},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        "departureTime": next_weekday_at_hour(hour),
    }
    data_bytes = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(ROUTES_URL, data=data_bytes, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-Goog-Api-Key", api_key)
    req.add_header("X-Goog-FieldMask", "routes.duration,routes.distanceMeters")

    with urllib.request.urlopen(req, timeout=10) as response:
        result = json.loads(response.read().decode())

    routes = result.get("routes")
    if not routes:
        return None

    duration_str = routes[0]["duration"]  # e.g. "312s"
    duration_sec = float(duration_str.rstrip("s"))
    distance_m = routes[0]["distanceMeters"]
    return {"duration_sec": duration_sec, "distance_m": distance_m}


def main():
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_MAPS_API_KEY environment variable not set (see geocode_stops.py docstring).")
        sys.exit(1)

    if not os.path.exists(COORDS_PATH):
        print(f"ERROR: {COORDS_PATH} not found. Run geocode_stops.py FIRST.")
        sys.exit(1)

    with open(COORDS_PATH) as f:
        coords = json.load(f)

    missing = [s for s in STOP_NAMES if s not in coords]
    if missing:
        print(f"ERROR: missing coordinates for: {missing}")
        print("Re-run geocode_stops.py and check it succeeded for all 18 stops.")
        sys.exit(1)

    n_segments = len(STOP_NAMES) - 1
    results = {}
    print(f"Fetching {n_segments} segments x {len(QUERY_HOURS)} times of day "
          f"= {n_segments * len(QUERY_HOURS)} API calls...")

    for i in range(n_segments):
        origin_name, dest_name = STOP_NAMES[i], STOP_NAMES[i + 1]
        segment_key = f"{origin_name} -> {dest_name}"
        results[segment_key] = {}
        print(f"[{i+1}/{n_segments}] {segment_key}")

        for hour in QUERY_HOURS:
            try:
                r = fetch_one_segment(coords[origin_name], coords[dest_name], hour, api_key)
            except Exception as e:
                print(f"    hour={hour}: ERROR {e}")
                r = None

            if r:
                print(f"    hour={hour:02d}:00 -> {r['duration_sec']/60:.1f} min, {r['distance_m']/1000:.2f} km")
                results[segment_key][str(hour)] = r
            time.sleep(0.2)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
