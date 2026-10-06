"""CrowdFlow decision-support dashboard for MTC Route 21G."""

import json
import os
import secrets
import sys

import pandas as pd
import pydeck as pdk
import streamlit as st

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIMULATOR_DIR = os.path.join(PROJECT_DIR, "simulator")
RL_DIR = os.path.join(PROJECT_DIR, "rl")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
EVALUATION_DIR = os.path.join(DATA_DIR, "dr90")
COORDS_PATH = os.path.join(DATA_DIR, "real_stop_coordinates.json")

sys.path.insert(0, SIMULATOR_DIR)
from route_model import ROUTE_CONFIG  # noqa: E402

sys.path.insert(0, RL_DIR)
from bus_env import BusDispatchEnv  # noqa: E402

STOP_NAMES = ROUTE_CONFIG["stop_names"]
DISPATCH_TIMES = ROUTE_CONFIG["real_dispatch_times_min"]

st.set_page_config(
    page_title="CrowdFlow | Route 21G",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #14283f;
        --muted: #61738a;
        --line: #dce5ed;
        --surface: #ffffff;
        --canvas: #f3f6f9;
        --navy: #102b46;
        --teal: #087e78;
        --amber: #9b5a08;
    }
    .stApp {background: var(--canvas); color: var(--ink) !important;}
    .stApp p, .stApp label, .stApp small, .stApp h1, .stApp h2,
    .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: var(--ink) !important;
        opacity: 1 !important;
    }
    .block-container {max-width: 1440px; padding: 1.6rem 2.2rem 3rem;}
    [data-testid="stHeader"] {background: rgba(243, 246, 249, .92);}
    [data-testid="stRadio"] > div {gap: .55rem;}
    [data-testid="stRadio"] label {
        border: 1px solid var(--line);
        border-radius: .65rem;
        background: var(--surface);
        padding: .35rem .8rem;
    }
    [data-testid="stRadio"] label, [data-testid="stRadio"] label * {
        color: var(--ink) !important;
        opacity: 1 !important;
    }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: .8rem;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 8px rgba(20, 40, 63, .035);
    }
    [data-testid="stMetric"] *,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
        opacity: 1 !important;
    }
    [data-testid="stAlert"] * {color: #35485d !important; opacity: 1 !important;}
    .brand-row {display:flex; align-items:center; gap:.8rem; margin-bottom:.15rem;}
    .brand-mark {
        display:flex; align-items:center; justify-content:center;
        width:2.5rem; height:2.5rem; border-radius:.7rem;
        background:var(--navy); color:#fff; font-size:1.25rem;
    }
    .brand-title {font-size:1.75rem; line-height:1.1; font-weight:750; color:var(--ink);}
    .brand-subtitle {color:var(--muted); margin:.35rem 0 1rem;}
    .status-pill {
        display:inline-flex; align-items:center; gap:.4rem;
        border:1px solid #ecd8b5; border-radius:999px; padding:.35rem .7rem;
        color:#71470c; background:#fff8e9; font-size:.78rem; font-weight:700;
    }
    .eyebrow {color:var(--teal); font-size:.74rem; font-weight:750;
        letter-spacing:.1em; text-transform:uppercase;}
    .hero {
        border-radius:1rem; padding:1.35rem 1.5rem;
        color:#f7fbff; background:linear-gradient(110deg, #102b46, #174768);
        margin:.5rem 0 1.2rem;
    }
    .hero h2 {color:#fff !important; margin:.25rem 0 .35rem; font-size:1.65rem;}
    .hero p {color:#d7e6f2 !important; margin:0; max-width:850px;}
    .section-note {color:var(--muted); margin-top:-.45rem; margin-bottom:1rem;}
    .small-note {font-size:.86rem; color:var(--muted);}
    div.stButton > button[kind="primary"] {
        background:#087e78; border-color:#087e78; border-radius:.65rem;
    }
    div.stButton > button[kind="primary"]:hover {
        background:#066a65; border-color:#066a65;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clock_time(minutes_after_midnight):
    total_minutes = int(round(minutes_after_midnight)) % (24 * 60)
    hour, minute = divmod(total_minutes, 60)
    suffix = "AM" if hour < 12 else "PM"
    return f"{hour % 12 or 12}:{minute:02d} {suffix}"


def read_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


@st.cache_data
def load_coordinates():
    if not os.path.exists(COORDS_PATH):
        return {}
    data = read_json(COORDS_PATH)
    return {
        name: {"lat": value["lat"], "lng": value["lng"]}
        for name, value in data.items()
    }


@st.cache_data
def load_lstm_evaluation():
    return read_json(os.path.join(EVALUATION_DIR, "lstm_eval.json"))


@st.cache_data
def load_planner_evaluation():
    return read_json(
        os.path.join(EVALUATION_DIR, "future_path_planner_evaluation.json")
    )


@st.cache_data(show_spinner=False)
def evaluate_departure(seed, bus_id):
    """Simulate prior scheduled buses, then assess one dispatch decision."""
    planner = BusDispatchEnv(seed=seed)
    planner.reset()

    while planner.bus_ptr + 1 < bus_id:
        _, _, terminated, _, _ = planner.depart_with_delay(0)
        if terminated:
            raise ValueError(f"Bus {bus_id} is outside the configured service day")

    plan = planner.plan_current_departure()
    best = plan["best_candidate"]
    scheduled_departure = planner.dispatch_times[planner.bus_ptr]
    candidate_rows = [
        {
            "Hold": f"{candidate['delay_seconds']} sec",
            "Departure time": clock_time(
                scheduled_departure + candidate["delay_seconds"] / 60
            ),
            "Route spacing": (
                "Meets modeled threshold"
                if candidate["is_safe"]
                else "Below modeled threshold"
            ),
            "Tightest stop": candidate["closest_stop_name"],
            "Minimum projected gap": f"{candidate['closest_gap_min']:.1f} min",
        }
        for candidate in sorted(
            plan["candidates"],
            key=lambda item: (
                item["max_unsafe_ratio"],
                item["mean_unsafe_ratio"],
                item["score"],
            ),
        )[:8]
    ]
    return {
        "bus_id": bus_id,
        "scheduled_departure_min": scheduled_departure,
        "recommended_delay_seconds": best["delay_seconds"],
        "recommended_departure_min": scheduled_departure + best["delay_seconds"] / 60,
        "closest_stop_name": best["closest_stop_name"],
        "closest_gap_min": best["closest_gap_min"],
        "scheduled_gap_min": best["scheduled_gap_min"],
        "minimum_safe_gap_min": best["minimum_safe_gap_min"],
        "is_safe": best["is_safe"],
        "warning": plan["warning"],
        "candidates": candidate_rows,
    }


def render_header():
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-mark">↗</div>
            <div class="brand-title">CrowdFlow</div>
        </div>
        <div class="brand-subtitle">Bus bunching prediction &amp; departure planning · MTC Route 21G</div>
        <span class="status-pill">● SIMULATION PROTOTYPE · NOT LIVE OPERATIONS</span>
        """,
        unsafe_allow_html=True,
    )


def render_operations():
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow" style="color:#8fe0d6">OPERATIONS WORKSPACE</div>
            <h2>From headway risk to a dispatch decision</h2>
            <p>Explore how a stochastic service-day scenario changes bus spacing, then compare possible terminal departure actions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "All generated service conditions are synthetic. Route and schedule "
        "inputs are project calibrations; passenger demand, traffic variation "
        "and incidents are modeled assumptions. This is not connected to live "
        "AVL, GTFS-realtime, or MTC dispatch systems."
    )

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Route stops", len(STOP_NAMES))
    kpi_cols[1].metric("Scheduled departures", len(DISPATCH_TIMES))
    kpi_cols[2].metric("Forecast horizon", "3 stops")
    kpi_cols[3].metric("Control point", "Origin terminal")
    st.markdown("### Departure review")
    st.markdown(
        '<div class="section-note">Choose one bus to replay its lead-in and assess candidate departure holds against the modeled route-spacing threshold.</div>',
        unsafe_allow_html=True,
    )

    bus_col, action_col, space_col = st.columns([1.1, 1.4, 2])
    with bus_col:
        bus_id = st.selectbox(
            "Scheduled bus",
            options=list(range(2, len(DISPATCH_TIMES) + 1)),
            format_func=lambda number: (
                f"Bus {number} · {clock_time(DISPATCH_TIMES[number - 1])}"
            ),
        )
    with action_col:
        st.write("")
        st.write("")
        generate = st.button(
            "Generate fresh service day",
            type="primary",
            help="Randomizes modeled traffic, incident and passenger conditions.",
        )
    if "scenario_seed" not in st.session_state:
        st.session_state.scenario_seed = secrets.randbits(32)
    if generate:
        st.session_state.scenario_seed = secrets.randbits(32)

    with st.spinner("Replaying lead-in buses and evaluating departure options…"):
        plan = evaluate_departure(st.session_state.scenario_seed, bus_id)
    with space_col:
        st.write("")
        st.caption(
            f"Scenario `SIM-{st.session_state.scenario_seed:08X}` · "
            "Prior buses use scheduled departures; selected bus is assessed "
            "against its lead bus."
        )

    st.divider()
    if plan["warning"]:
        st.warning(plan["warning"])
    elif not plan["is_safe"]:
        st.warning(
            "No candidate meets the model's spacing threshold. Treat this as "
            "a simulation result requiring review, not an instruction to hold."
        )
    elif plan["recommended_delay_seconds"] == 0:
        st.success(
            "Scenario recommendation: no hold is needed to meet the modeled "
            "spacing threshold."
        )
    else:
        st.success(
            f"Scenario recommendation: hold Bus {bus_id} for "
            f"{plan['recommended_delay_seconds']} seconds. The modeled "
            "departure is "
            f"{clock_time(plan['recommended_departure_min'])}."
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Scheduled departure", clock_time(plan["scheduled_departure_min"])
    )
    metric_cols[1].metric(
        "Suggested hold", f"{plan['recommended_delay_seconds']} sec"
    )
    metric_cols[2].metric(
        "Narrowest projected gap", f"{plan['closest_gap_min']:.1f} min"
    )
    metric_cols[3].metric("Tightest route point", plan["closest_stop_name"])

    st.markdown("#### Candidate comparison")
    st.caption(
        "The safety label applies only to the simulator's 50%-of-scheduled-gap "
        "rule. It is not a guarantee of safe service or passenger benefit."
    )
    st.dataframe(
        pd.DataFrame(plan["candidates"]),
        width="stretch",
        hide_index=True,
    )

    with st.expander("What this scenario does—and does not—represent"):
        st.markdown(
            "- The selected bus is evaluated after earlier buses in the same randomized run depart on schedule.\n"
            "- Each candidate replays the selected bus over the modeled route with shared day-level traffic, passenger and incident assumptions.\n"
            "- The planner assesses spacing relative to the bus ahead. It is not the PPO policy and does not currently consume LSTM output as its state.\n"
            "- No live bus positions, passenger counts, incidents or traffic feeds are connected."
        )

    st.markdown("### CrowdFlow workflow")
    workflow_cols = st.columns(3)
    workflow = [
        (
            "01 · SIMULATE",
            "Model the service day",
            "Route schedule, passenger accumulation, dwell time, traffic variation and incidents form the experimental operating conditions.",
        ),
        (
            "02 · PREDICT",
            "Estimate bunching risk",
            "A dual-head LSTM forecasts headway deviation and Low / Medium / High risk up to three stops ahead from recent stop events.",
        ),
        (
            "03 · EVALUATE",
            "Compare interventions",
            "PPO and deterministic departure planning are evaluated against no-delay and fixed-rule strategies; this page's interactive planner is a separate comparison tool.",
        ),
    ]
    for column, (step, title, description) in zip(workflow_cols, workflow):
        with column:
            with st.container(border=True):
                st.markdown(f'<div class="eyebrow">{step}</div>', unsafe_allow_html=True)
                st.markdown(f"**{title}**")
                st.caption(description)


def render_rider_workspace():
    st.markdown("### Route guide")
    st.markdown(
        '<div class="section-note">Plan a section of the 21G route and check the published departure pattern. Arrival estimates and live vehicle positions are not available.</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Static route reference. Departure times shown are the project's "
        "configured schedule inputs; check MTC for current service changes."
    )
    origin_col, destination_col = st.columns(2)
    with origin_col:
        origin = st.selectbox(
            "Board at",
            STOP_NAMES[:-1],
            format_func=lambda stop: f"{STOP_NAMES.index(stop) + 1:02d}  ·  {stop}",
        )
    origin_index = STOP_NAMES.index(origin)
    destination_options = STOP_NAMES[origin_index + 1 :]
    with destination_col:
        destination = st.selectbox(
            "Get off at",
            destination_options,
            index=min(4, len(destination_options) - 1),
            format_func=lambda stop: (
                f"{STOP_NAMES.index(stop) + 1:02d}  ·  {stop}"
            ),
        )

    destination_index = STOP_NAMES.index(destination)
    journey_stops = STOP_NAMES[origin_index : destination_index + 1]
    metric_cols = st.columns(3)
    metric_cols[0].metric("Board", origin)
    metric_cols[1].metric("Alight", destination)
    metric_cols[2].metric("Stops including endpoints", len(journey_stops))

    map_col, list_col = st.columns([1.15, 1])
    coordinates = load_coordinates()
    mapped = [
        {
            "name": stop,
            "lat": coordinates[stop]["lat"],
            "lon": coordinates[stop]["lng"],
            "sequence": index + 1,
            "color": [8, 126, 120]
            if stop in (origin, destination)
            else [49, 100, 151],
        }
        for index, stop in enumerate(journey_stops, start=origin_index)
        if stop in coordinates
    ]
    with map_col:
        st.markdown("#### Selected route section")
        if len(mapped) >= 2:
            path = [[stop["lon"], stop["lat"]] for stop in mapped]
            map_view = pdk.ViewState(
                latitude=sum(stop["lat"] for stop in mapped) / len(mapped),
                longitude=sum(stop["lon"] for stop in mapped) / len(mapped),
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
                            data=mapped,
                            get_position=["lon", "lat"],
                            get_fill_color="color",
                            get_radius=110,
                            pickable=True,
                        ),
                    ],
                    initial_view_state=map_view,
                    tooltip={"text": "{sequence}. {name}"},
                    map_style="road",
                ),
                width="stretch",
            )
        else:
            st.warning("Map coordinates are unavailable for this route section.")

    with list_col:
        st.markdown("#### Stops in order")
        st.dataframe(
            pd.DataFrame(
                {
                    "No.": range(origin_index + 1, destination_index + 2),
                    "Stop": journey_stops,
                    "Action": [
                        "Board" if stop == origin
                        else "Alight" if stop == destination
                        else ""
                        for stop in journey_stops
                    ],
                }
            ),
            width="stretch",
            hide_index=True,
            height=min(500, 38 * len(journey_stops) + 40),
        )

    with st.expander("Configured departures from Parrys Corner"):
        departures = pd.DataFrame(
            {
                "Bus": range(1, len(DISPATCH_TIMES) + 1),
                "Scheduled departure": [
                    clock_time(value) for value in DISPATCH_TIMES
                ],
            }
        )
        st.dataframe(departures, width="stretch", hide_index=True)
        st.caption(
            "This is a static project timetable input, not a live departure board."
        )

    with st.expander("Browse all 47 stops"):
        query = st.text_input("Search route stops", placeholder="e.g. Guindy")
        directory = pd.DataFrame(
            {"Stop": range(1, len(STOP_NAMES) + 1), "Name": STOP_NAMES}
        )
        if query:
            directory = directory[
                directory["Name"].str.contains(query, case=False, regex=False)
            ]
        st.dataframe(directory, width="stretch", hide_index=True)


def render_evaluation():
    st.markdown("### Model & experiment results")
    st.markdown(
        '<div class="section-note">Reported project evaluation artifacts on generated synthetic data—not current-day predictions or field results.</div>',
        unsafe_allow_html=True,
    )
    st.warning(
        "These figures describe offline experiments. They should not be "
        "interpreted as live-route accuracy or evidence of deployed "
        "operational impact."
    )
    lstm = load_lstm_evaluation()
    regression = pd.DataFrame(lstm["regression"])
    classification = lstm["classification"]

    score_cols = st.columns(3)
    score_cols[0].metric(
        "Classification accuracy",
        f"{classification['lstm_accuracy']:.1%}",
        help="LSTM accuracy on the project's held-out synthetic evaluation set.",
    )
    score_cols[1].metric(
        "High-risk recall",
        f"{classification['lstm_high_risk_recall']:.1%}",
        help="Share of high-risk labels recovered in offline synthetic evaluation.",
    )
    horizon_three = regression[regression["horizon"] == 3].iloc[0]
    score_cols[2].metric(
        "3-stop forecast MAE",
        f"{horizon_three['lstm_mae']:.2f} min",
        help="Mean absolute error for predicted headway deviation.",
    )

    st.markdown("#### Prediction benchmark")
    prediction_table = pd.DataFrame(
        {
            "Forecast horizon": [
                f"{int(row.horizon)} stop"
                if row.horizon == 1
                else f"{int(row.horizon)} stops"
                for row in regression.itertuples(index=False)
            ],
            "LSTM deviation MAE": [
                f"{value:.3f} min" for value in regression["lstm_mae"]
            ],
            "Persistence baseline MAE": [
                f"{value:.3f} min" for value in regression["persistence_mae"]
            ],
            "Improvement vs baseline": [
                f"{value:.1f}%" for value in regression["improvement_pct"]
            ],
        }
    )
    st.dataframe(prediction_table, width="stretch", hide_index=True)
    st.caption(
        f"High-risk recall baseline: persistence "
        f"{classification['persistence_high_risk_recall']:.1%} · "
        f"Always-low accuracy baseline: "
        f"{classification['always_low_accuracy']:.1%}."
    )

    planner_results = load_planner_evaluation()
    benchmark_names = [
        "No departure delay",
        "Fixed departure rule",
        "Future-path departure planner",
    ]
    benchmark_rows = [
        {
            "Strategy": name,
            "Mean headway deviation ratio": (
                planner_results[name]["mean_ahead_ratio"]
            ),
            "Departures flagged high-risk": (
                planner_results[name]["pct_departures_high_risk"]
            ),
            "Mean departure delay": (
                planner_results[name]["mean_departure_delay_seconds"]
            ),
        }
        for name in benchmark_names
    ]
    benchmark = pd.DataFrame(benchmark_rows)
    st.markdown("#### Departure-planning benchmark")
    st.dataframe(
        benchmark.style.format(
            {
                "Mean headway deviation ratio": "{:.3f}",
                "Departures flagged high-risk": "{:.1f}%",
                "Mean departure delay": "{:.2f} sec",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    improvement = planner_results["planner_ahead_improvement_pct_vs_no_delay"]
    st.caption(
        f"In the stored experiment, the future-path planner's mean ahead-bus "
        f"deviation ratio improved by {improvement:.1f}% vs. no delay. This is "
        "a simulation benchmark and should be considered with the full "
        "evaluation protocol and limitations."
    )

    with st.expander("System scope and limitations"):
        st.markdown(
            "- **Route model:** 47-stop representation; much of the fine-grained travel-time profile is interpolated, and the Kilambakkam extension is estimated.\n"
            "- **Demand and traffic:** passenger arrivals, dwell noise, traffic variation and incidents are simulated assumptions.\n"
            "- **Prediction:** the LSTM predicts headway deviation and risk from historical stop-event sequences. This dashboard shows offline evaluation metrics, not live model inference.\n"
            "- **Control:** the PPO agent is a separate experimental policy. The interactive future-path planner is not the PPO agent and does not consume an LSTM prediction.\n"
            "- **Deployment:** there is no live AVL/GPS or GTFS-realtime feed; no recommendation is sent to a bus or dispatcher."
        )


render_header()
workspace = st.radio(
    "Workspace",
    ["Operations", "Rider route guide", "Model evaluation"],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()

if workspace == "Operations":
    render_operations()
elif workspace == "Rider route guide":
    render_rider_workspace()
else:
    render_evaluation()
