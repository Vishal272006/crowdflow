"""CrowdFlow rider route guide and bus dispatch planning dashboard."""

import json
import os
import sys

import pandas as pd
import pydeck as pdk
import streamlit as st

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIMULATOR_DIR = os.path.join(PROJECT_DIR, "simulator")
RL_DIR = os.path.join(PROJECT_DIR, "rl")
sys.path.insert(0, SIMULATOR_DIR)
from route_model import ROUTE_CONFIG  # noqa: E402

sys.path.insert(0, RL_DIR)
from bus_env import BusDispatchEnv  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
COORDS_PATH = os.path.join(DATA_DIR, "real_stop_coordinates.json")
STOP_NAMES = ROUTE_CONFIG["stop_names"]

st.set_page_config(page_title="CrowdFlow | Route 21G", page_icon="🚌", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: #f4f7fa;
        border: 1px solid #e1e8ef;
        padding: 1rem;
        border-radius: 0.75rem;
    }
    .eyebrow {color: #52667a; font-size: .78rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase;}
    </style>
    """,
    unsafe_allow_html=True,
)


def clock_time(minutes_after_midnight):
    total_minutes = int(round(minutes_after_midnight)) % (24 * 60)
    hour, minute = divmod(total_minutes, 60)
    suffix = "AM" if hour < 12 else "PM"
    return f"{hour % 12 or 12}:{minute:02d} {suffix}"


@st.cache_data
def load_real_coordinates():
    if not os.path.exists(COORDS_PATH):
        return {}
    with open(COORDS_PATH, encoding="utf-8") as coordinates_file:
        data = json.load(coordinates_file)
    return {
        name: {"lat": value["lat"], "lng": value["lng"]}
        for name, value in data.items()
    }


@st.cache_data(show_spinner="Building an example service-day plan...")
def load_planner_scenario(seed=999):
    """Return advisory dispatch choices for one generated service-day scenario."""
    planner = BusDispatchEnv(seed=seed)
    planner.reset()
    decisions = {}
    terminated = False
    while not terminated:
        bus_id = planner.bus_ptr + 1
        plan = planner.plan_current_departure()
        best = plan["best_candidate"]
        scheduled_departure_min = planner.dispatch_times[planner.bus_ptr]
        decisions[bus_id] = {
            "recommended_delay_seconds": best["delay_seconds"],
            "scheduled_departure_min": scheduled_departure_min,
            "recommended_departure_min": (
                scheduled_departure_min + best["delay_seconds"] / 60.0
            ),
            "closest_stop_name": best["closest_stop_name"],
            "closest_gap_min": best["closest_gap_min"],
            "scheduled_gap_min": best["scheduled_gap_min"],
            "minimum_safe_gap_min": best["minimum_safe_gap_min"],
            "is_safe": best["is_safe"],
            "warning": plan["warning"],
            "candidates": [
                {
                    "Departure delay": (
                        f"{candidate['delay_seconds']} sec"
                    ),
                    "Suggested departure": clock_time(
                        scheduled_departure_min
                        + candidate["delay_seconds"] / 60.0
                    ),
                    "Projected route safety": (
                        "Within modeled threshold"
                        if candidate["is_safe"]
                        else "Below modeled threshold"
                    ),
                    "Closest projected stop": candidate["closest_stop_name"],
                    "Projected gap": (
                        f"{candidate['closest_gap_min']:.1f} min"
                    ),
                }
                for candidate in sorted(
                    plan["candidates"],
                    key=lambda item: (
                        item["max_unsafe_ratio"],
                        item["mean_unsafe_ratio"],
                        item["score"],
                    ),
                )[:5]
            ],
        }
        _, _, terminated, _, _ = planner.depart_with_delay(best["delay_seconds"])
    return decisions


def render_rider_guide():
    st.subheader("Plan a ride")
    st.write(
        "Choose your boarding and alighting stops to see the stops in between "
        "and follow that section of the route."
    )
    st.info(
        "Route information only: this dashboard does not receive live bus "
        "locations, current departures, or real-time arrival estimates."
    )

    origin_col, destination_col = st.columns(2)
    with origin_col:
        origin = st.selectbox(
            "Board at",
            STOP_NAMES[:-1],
            format_func=lambda name: f"{STOP_NAMES.index(name) + 1}. {name}",
        )
    origin_index = STOP_NAMES.index(origin)
    destination_options = STOP_NAMES[origin_index + 1 :]
    with destination_col:
        destination = st.selectbox(
            "Get off at",
            destination_options,
            index=min(4, len(destination_options) - 1),
            format_func=lambda name: f"{STOP_NAMES.index(name) + 1}. {name}",
        )

    destination_index = STOP_NAMES.index(destination)
    journey_stops = STOP_NAMES[origin_index : destination_index + 1]
    first_metric, second_metric, third_metric = st.columns(3)
    first_metric.metric("Board at", origin)
    second_metric.metric("Get off at", destination)
    third_metric.metric("Stops on this section", len(journey_stops))

    map_col, stops_col = st.columns([1.2, 1])
    coordinates = load_real_coordinates()
    with map_col:
        st.markdown("#### Your section of Route 21G")
        mapped_stops = [
            {
                "name": name,
                "lat": coordinates[name]["lat"],
                "lon": coordinates[name]["lng"],
                "position": index + 1,
                "color": (
                    [20, 117, 96]
                    if name in (origin, destination)
                    else [49, 100, 151]
                ),
            }
            for index, name in enumerate(journey_stops, start=origin_index)
            if name in coordinates
        ]
        if len(mapped_stops) >= 2:
            path = [[stop["lon"], stop["lat"]] for stop in mapped_stops]
            view = pdk.ViewState(
                latitude=sum(stop["lat"] for stop in mapped_stops)
                / len(mapped_stops),
                longitude=sum(stop["lon"] for stop in mapped_stops)
                / len(mapped_stops),
                zoom=10,
            )
            st.pydeck_chart(
                pdk.Deck(
                    layers=[
                        pdk.Layer(
                            "PathLayer",
                            data=[{"path": path}],
                            get_path="path",
                            get_color=[49, 100, 151],
                            width_min_pixels=4,
                        ),
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=mapped_stops,
                            get_position=["lon", "lat"],
                            get_fill_color="color",
                            get_radius=110,
                            pickable=True,
                        ),
                    ],
                    initial_view_state=view,
                    tooltip={"text": "{position}. {name}"},
                    map_style="road",
                ),
                width="stretch",
            )
        else:
            st.warning("Map coordinates are not available for this section.")

    with stops_col:
        st.markdown("#### Stops in order")
        journey_table = pd.DataFrame(
            {
                "Stop": range(origin_index + 1, destination_index + 2),
                "Name": journey_stops,
                "Your stop": [
                    "Board" if name == origin else "Get off" if name == destination else ""
                    for name in journey_stops
                ],
            }
        )
        st.dataframe(
            journey_table,
            width="stretch",
            hide_index=True,
            height=min(500, 38 * len(journey_stops) + 40),
        )

    with st.expander("Browse all stops on this direction"):
        search = st.text_input("Find a stop", placeholder="Type a stop name")
        directory = pd.DataFrame(
            {
                "Stop": range(1, len(STOP_NAMES) + 1),
                "Name": STOP_NAMES,
                "Direction": "Parrys Corner to Kilambakkam",
            }
        )
        if search:
            directory = directory[
                directory["Name"].str.contains(search, case=False, regex=False)
            ]
        st.dataframe(directory, width="stretch", hide_index=True)


def render_dispatch_planner():
    st.subheader("Dispatch planning")
    st.write(
        "Review a modeled departure recommendation and compare it with "
        "alternative start times."
    )
    st.warning(
        "Planning aid, not a live control system. These results are generated "
        "from a simulated service day; they do not use current MTC bus "
        "positions or traffic conditions. Verify against operations before acting."
    )

    scenario = load_planner_scenario()
    bus_ids = sorted(scenario)
    bus_id = st.selectbox(
        "Bus in the example service day",
        bus_ids,
        format_func=lambda number: f"Bus {number}",
    )
    plan = scenario[bus_id]

    st.markdown("#### Recommended action")
    if plan["warning"]:
        st.warning(plan["warning"])
    elif not plan["is_safe"]:
        st.warning(
            "The model did not find a departure that meets its route-spacing "
            "threshold. Escalate this case for human review."
        )
    elif plan["recommended_delay_seconds"] == 0:
        st.success("Model recommendation: depart at the scheduled time.")
    else:
        st.success(
            f"Model recommendation: hold for "
            f"{plan['recommended_delay_seconds']} seconds, then depart at "
            f"{clock_time(plan['recommended_departure_min'])}."
        )

    action_col, gap_col, stop_col = st.columns(3)
    action_col.metric(
        "Scheduled departure",
        clock_time(plan["scheduled_departure_min"]),
    )
    gap_col.metric(
        "Closest projected gap",
        f"{plan['closest_gap_min']:.1f} min",
        help="Smallest modeled arrival gap between this bus and the bus ahead.",
    )
    stop_col.metric("Tightest point on route", plan["closest_stop_name"])

    st.markdown("#### Compare departure options")
    st.caption(
        "Projected gaps and safety labels come from the simulation, not "
        "observed service data. The threshold is a planning assumption."
    )
    st.dataframe(
        pd.DataFrame(plan["candidates"]),
        width="stretch",
        hide_index=True,
    )

    with st.expander("How to interpret this recommendation"):
        st.markdown(
            "- **Scheduled departure** is the model's timetable for this example day.\n"
            "- **Closest projected gap** is the narrowest spacing the simulation predicts along the route.\n"
            "- **Within modeled threshold** means the simulation meets its configured minimum-gap rule; it is not a guarantee of safe or on-time operation.\n"
            "- The scenario uses estimated passenger demand and travel-time assumptions. Replace these with validated operational data before using recommendations in service."
        )


st.title("CrowdFlow")
st.caption("Route 21G · Parrys Corner to Kilambakkam, Chennai")

audience = st.radio(
    "Choose a workspace",
    ["Rider route guide", "Dispatch planning"],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()

if audience == "Rider route guide":
    render_rider_guide()
else:
    render_dispatch_planner()
