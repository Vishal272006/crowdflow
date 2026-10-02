
"""
geocode_stops.py

Geocode all 47 MTC Route 21G stops to latitude/longitude.

Source of stop names:
    simulator/route_model.py -> ROUTE_CONFIG["stop_names"]

Geocoding service:
    OpenStreetMap Nominatim

Output:
    data/real_stop_coordinates.json

IMPORTANT:
    - Nominatim is rate-limited. We wait 1.1 seconds between requests.
    - Successful coordinates are cached.
    - If the script is interrupted, simply run it again.
    - Already-successful stops will NOT be queried again.
    - ALWAYS inspect matched_address before trusting the coordinates.
"""

import json
import os
import sys
import time

import requests


# ---------------------------------------------------------------------------
# IMPORT ROUTE CONFIGURATION
# ---------------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "simulator")
)

from route_model import ROUTE_CONFIG  # noqa: E402


# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

USER_AGENT = (
    "crowdflow-capstone-project "
    "(student project, MTC 21G bunching prediction)"
)

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "real_stop_coordinates.json"
)

# Nominatim public server:
# Keep requests at least 1 second apart.
RATE_LIMIT_SECONDS = 1.1


# ---------------------------------------------------------------------------
# SEARCH OVERRIDES
#
# The LEFT side must exactly match the stop name in ROUTE_CONFIG.
#
# The RIGHT side contains alternative search phrases.
# They are attempted from top to bottom.
# ---------------------------------------------------------------------------

QUERY_OVERRIDES = {

    # -----------------------------------------------------------------------
    # NORTH / CENTRAL CHENNAI
    # -----------------------------------------------------------------------

    "R B I Parrys": [
        "Reserve Bank of India, Parrys Corner, Chennai",
        "RBI Chennai",
        "Reserve Bank of India Chennai",
    ],

    "Kannagi Statue or Presidency College": [
        "Presidency College, Chennai",
        "Kannagi Statue, Marina Beach, Chennai",
    ],

    "Vivekananda House Bus Terminus": [
        "Swami Vivekananda House, Chennai",
        "Vivekananda House, Triplicane, Chennai",
        "Vivekananda House, Marina Beach, Chennai",
    ],

    "Triplicane Queen Marys College": [
        "Queen Mary's College, Chennai",
        "Queen Marys College, Chennai",
        "Queen Mary's College, Mylapore, Chennai",
    ],

    "City Centre or Kalyani Hospital": [
        "Kalyani Hospital, Royapettah, Chennai",
        "Kalyani Hospital, Chennai",
        "Royapettah, Chennai",
    ],

    "Royapettah Yellow Pages": [
        "Royapettah, Chennai",
        "Yellow Pages Royapettah, Chennai",
    ],

    "DGP Office": [
        "DGP Office, Chennai",
        "Director General of Police Office, Chennai",
        "DGP Office, Mylapore, Chennai",
        "DGP Office, Dr Radhakrishnan Salai, Chennai",
    ],

    "CLRI or IIT Madras": [
        "IIT Madras, Chennai",
        "Indian Institute of Technology Madras, Chennai",
        "Central Leather Research Institute, Chennai",
        "CLRI Chennai",
    ],

    "B M Birla Planetarium": [
        "B M Birla Planetarium, Chennai",
        "Birla Planetarium, Chennai",
        "B.M. Birla Planetarium, Kotturpuram, Chennai",
        "Birla Planetarium Kotturpuram, Chennai",
    ],


    # -----------------------------------------------------------------------
    # SAIDAPET / GUINDY / ALANDUR
    # -----------------------------------------------------------------------

    "Saidapet Court or Saidapet Depot": [
        "Saidapet Court, Chennai",
        "Saidapet Court Chennai",
        "Saidapet, Chennai",
    ],

    "Guindy B.T": [
        "Guindy Bus Terminus, Chennai",
        "Guindy Bus Stand, Chennai",
        "Guindy, Chennai",
    ],

    "Alandur Metro R.S": [
        "Alandur Metro Station, Chennai",
        "Alandur Metro, Chennai",
        "Alandur, Chennai",
    ],

    "St. Thomas Mount P.O": [
        "St Thomas Mount Post Office, Chennai",
        "St Thomas Mount Head Post Office, Chennai",
        "St Thomas Mount, Chennai",
    ],


    # -----------------------------------------------------------------------
    # AIRPORT / PALLAVARAM / CHROMEPET
    # -----------------------------------------------------------------------

    "Meenambakkam International Airport": [
        "Meenambakkam, Chennai",
        "Chennai International Airport, Meenambakkam",
        "Meenambakkam Airport, Chennai",
        "Chennai Airport, Chennai",
    ],

    "Thirusoolam National Airport": [
        "Tirusulam, Chennai",
        "Trisulam, Chennai",
        "Tirusulam Airport, Chennai",
        "Chennai International Airport, Tirusulam",
    ],

    "Chromepet Ponds Company": [
        "Ponds Factory, Chromepet, Chennai",
        "Ponds India Chromepet, Chennai",
        "Chromepet, Chennai",
    ],

    "Saravana Store Chromepet": [
        "Saravana Stores, Chromepet, Chennai",
        "Super Saravana Stores, Chromepet, Chennai",
        "Saravana Store Chromepet, Chennai",
        "Chromepet, Chennai",
    ],

    "Chromepet MIT Gate": [
        "MIT Gate, Chromepet, Chennai",
        "MIT Main Gate, Chromepet, Chennai",
        "Madras Institute of Technology, Chromepet, Chennai",
        "MIT Chromepet, Chennai",
    ],


    # -----------------------------------------------------------------------
    # TAMBARAM / SANATORIUM
    # -----------------------------------------------------------------------

    "Tambaram TB Hospital": [
        "Government Hospital for Thoracic Medicine, Tambaram",
        "Government Hospital of Thoracic Medicine, Tambaram Sanatorium",
        "TB Hospital, Tambaram Sanatorium, Chennai",
        "TB Hospital Tambaram, Chennai",
        "Tambaram Sanatorium, Chennai",
    ],

    "Tambaram Sanatorium B.T": [
        "Tambaram Sanatorium Bus Stand, Chennai",
        "Sanatorium Bus Stand, Tambaram, Chennai",
        "Tambaram Sanatorium, Chennai",
        "Sanatorium, Tambaram, Chennai",
    ],

    "Kadaperi": [
        "Kadaperi, Tambaram, Chennai",
        "Kadaperi, Chennai",
        "Kadaperi Bus Stop, Chennai",
    ],

    "Tambaram West Bus Stand": [
        "Tambaram West Bus Stand, Chennai",
        "West Tambaram Bus Stand, Chennai",
        "Tambaram West, Chennai",
    ],


    # -----------------------------------------------------------------------
    # PERUNGALATHUR / VANDALUR
    # -----------------------------------------------------------------------

    "Perungalatgur Lake View Stop": [
        "Perungalattur Lake View, Chennai",
        "Perungalathur Lake, Chennai",
        "Perungalattur Lake, Chennai",
        "Perungalathur, Chennai",
    ],

    "Perungalattur": [
        "Perungalattur, Chennai",
        "Perungalathur, Chennai",
        "Perungalattur Bus Stop, Chennai",
        "Perungalathur Bus Stop, Chennai",
    ],

    "Perungalathur Iraniyamman Temple": [
        "Iraniyamman Temple, Perungalathur, Chennai",
        "Iraniyamman Temple, Vandalur, Chennai",
        "Arulmigu Iraniyamman Temple, Chennai",
        "Iraniyamman Temple, Perungalattur, Chennai",
    ],

    "Vandalur Gate": [
        "Vandalur Gate, Chennai",
        "Vandalur Zoo entrance, Chennai",
        "Vandalur, Chennai",
    ],

    "Vandalur Zoo": [
        "Vandalur Zoo, Chennai",
        "Arignar Anna Zoological Park, Vandalur",
        "Vandalur Zoo Bus Stop, Chennai",
        "Arignar Anna Zoological Park Bus Stop",
    ],


    # -----------------------------------------------------------------------
    # KILAMBAKKAM
    # -----------------------------------------------------------------------

    "Kilambakkam Bus Terminus": [
        "Kilambakkam Bus Terminus, Chennai",
        "Kilambakkam New Bus Stand, Chennai",
        "Kalaignar Centenary Bus Terminus, Kilambakkam",
        "Kilambakkam Bus Stand, Chennai",
    ],
}


# ---------------------------------------------------------------------------
# MANUAL COORDINATES
#
# ONLY USE THIS IF ALL SEARCH QUERIES FAIL.
#
# Format:
#
# "Exact stop name": (latitude, longitude)
#
# IMPORTANT:
# Do NOT put approximate/random coordinates here.
# Use the actual bus-stop/terminal location.
# ---------------------------------------------------------------------------

MANUAL_COORDINATES = {
     "Vivekananda House Bus Terminus": (13.054520, 80.274223),
    "DGP Office": (13.0456, 80.2778),
    "Chromepet MIT Gate": (12.943968, 80.160025),
    "Kadaperi": (12.924826, 80.117188),
    "Perungalathur Iraniyamman Temple": (12.9056, 80.0953),
    "Vandalur Zoo": (12.88273, 80.08183),
    "Kilambakkam Bus Terminus": (12.871897, 80.080828),
}


# ---------------------------------------------------------------------------
# GEOCODE ONE LOCATION
# ---------------------------------------------------------------------------

def geocode_one(
    query_text,
    city_hint="Chennai, Tamil Nadu, India"
):
    """
    Query Nominatim for one search string.

    Returns:
        {
            "lat": float,
            "lng": float,
            "matched_address": str
        }

    Returns None if no result is found.
    """

    query = f"{query_text}, {city_hint}"

    response = requests.get(
        NOMINATIM_URL,
        params={
            "q": query,
            "format": "json",
            "limit": 1,
        },
        headers={
            "User-Agent": USER_AGENT
        },
        timeout=15,
    )

    response.raise_for_status()

    results = response.json()

    if not results:
        return None

    top = results[0]

    return {
        "lat": float(top["lat"]),
        "lng": float(top["lon"]),
        "matched_address": top.get("display_name", ""),
    }


# ---------------------------------------------------------------------------
# MAIN GEOCODING FUNCTION
# ---------------------------------------------------------------------------

def geocode_all_stops(
    config=ROUTE_CONFIG,
    output_path=OUTPUT_PATH
):

    stop_names = config["stop_names"]

    print("=" * 70)
    print("CROWDFLOW - MTC ROUTE 21G STOP GEOCODING")
    print("=" * 70)

    print(f"\nStops expected from ROUTE_CONFIG: {len(stop_names)}")

    # -----------------------------------------------------------------------
    # LOAD EXISTING RESULTS
    # -----------------------------------------------------------------------

    coords = {}

    if os.path.exists(output_path):

        print(f"\nExisting coordinate file found:")
        print(output_path)

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                coords = json.load(f)

        except (json.JSONDecodeError, OSError) as e:

            print(f"WARNING: Could not read existing file: {e}")
            print("Starting with an empty coordinate dictionary.\n")
            coords = {}

    already_ok = [
        name for name in stop_names
        if name in coords
    ]

    if already_ok:

        print(
            f"\nAlready geocoded: "
            f"{len(already_ok)}/{len(stop_names)}"
        )

        print("These stops will be skipped.")

    # -----------------------------------------------------------------------
    # DETERMINE WHAT STILL NEEDS TO BE QUERIED
    # -----------------------------------------------------------------------

    to_query = [
        name for name in stop_names
        if name not in coords
    ]

    print(
        f"\nRemaining stops to geocode: "
        f"{len(to_query)}"
    )

    if not to_query:
        print("\nAll stops are already present in the JSON file.")
        print(f"Output: {output_path}")
        return coords

    failures = []

    # -----------------------------------------------------------------------
    # GEOCODE EACH MISSING STOP
    # -----------------------------------------------------------------------

    for index, name in enumerate(to_query, start=1):

        candidates = QUERY_OVERRIDES.get(
            name,
            [name]
        )

        result = None

        print("\n" + "-" * 70)

        print(
            f"[{index}/{len(to_query)}] "
            f"STOP: {name}"
        )

        print(
            f"Candidate queries: {len(candidates)}"
        )

        # ---------------------------------------------------------------
        # TRY ALL SEARCH PHRASES
        # ---------------------------------------------------------------

        for candidate_index, query_text in enumerate(
            candidates,
            start=1
        ):

            print(
                f"\n  Try {candidate_index}/{len(candidates)}:"
                f" {query_text}"
            )

            try:

                result = geocode_one(query_text)

            except requests.RequestException as error:

                print(
                    f"  Request failed: {error}"
                )

                # Wait before another request.
                time.sleep(RATE_LIMIT_SECONDS)

                continue

            if result is None:

                print("  No result.")

                time.sleep(RATE_LIMIT_SECONDS)

                continue

            # -----------------------------------------------------------
            # SUCCESS
            # -----------------------------------------------------------

            print("  SUCCESS")

            print(
                f"  Coordinates:"
                f" {result['lat']}, {result['lng']}"
            )

            print(
                f"  Matched address:"
                f" {result['matched_address']}"
            )

            coords[name] = result

            time.sleep(RATE_LIMIT_SECONDS)

            break

        # -------------------------------------------------------------------
        # MANUAL FALLBACK
        # -------------------------------------------------------------------

        if result is None and name in MANUAL_COORDINATES:

            lat, lng = MANUAL_COORDINATES[name]

            result = {
                "lat": float(lat),
                "lng": float(lng),
                "matched_address": "(manually entered)"
            }

            coords[name] = result

            print("\n  MANUAL COORDINATE USED")

            print(
                f"  Coordinates: {lat}, {lng}"
            )

        # -------------------------------------------------------------------
        # COMPLETE FAILURE
        # -------------------------------------------------------------------

        if result is None:

            failures.append(name)

            print(
                "\n  FAILED: No coordinate found."
            )

    # -----------------------------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------------------------

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            coords,
            f,
            indent=2,
            ensure_ascii=False
        )

    # -----------------------------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GEOCODING COMPLETE")
    print("=" * 70)

    print(
        f"\nSuccessfully geocoded:"
        f" {len(coords)}/{len(stop_names)}"
    )

    print(
        f"Failed:"
        f" {len(failures)}/{len(stop_names)}"
    )

    print(
        f"\nOutput file:"
        f"\n{os.path.abspath(output_path)}"
    )

    # -----------------------------------------------------------------------
    # LIST FAILURES
    # -----------------------------------------------------------------------

    if failures:

        print("\n" + "=" * 70)
        print("STOPS STILL REQUIRING ATTENTION")
        print("=" * 70)

        for stop in failures:
            print(f"  - {stop}")

        print(
            "\nThese stops were NOT given fake coordinates."
        )

        print(
            "Add verified coordinates to MANUAL_COORDINATES "
            "or add better search queries to QUERY_OVERRIDES."
        )

    else:

        print("\n" + "=" * 70)
        print("SUCCESS: ALL STOPS HAVE COORDINATES")
        print("=" * 70)

    # -----------------------------------------------------------------------
    # IMPORTANT VERIFICATION WARNING
    # -----------------------------------------------------------------------

    print("\nIMPORTANT:")
    print(
        "A successful Nominatim result does not automatically mean "
        "the coordinate is the exact physical bus-stop location."
    )

    print(
        "Review the printed matched_address values, especially for "
        "landmark-based or ambiguous stop names."
    )

    return coords


# ---------------------------------------------------------------------------
# PROGRAM ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geocode_all_stops()
