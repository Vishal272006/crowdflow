"""
app.py -- CrowdFlow Dashboard

A Streamlit dashboard that walks through a held-out simulated bus trip
(sampled from val.csv, i.e. data the models never trained on) and shows,
stop by stop:
    - the actual headway deviation that happened
    - the LSTM's prediction, made using ONLY the stops seen so far
      (this is the honest, causally-correct way to show it -- no peeking
      at future stops)
    - the resulting bunching risk classification
    - the RL agent's one-time departure recommendation at Parrys Corner

Run with:
    streamlit run app.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pydeck as pdk
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rl"))
from inference import get_engine, FEATURE_COLS  # noqa: E402
from bus_env import BusDispatchEnv, load_trips, N_STOPS  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SIMULATION_DATA_DIR = os.path.join(DATA_DIR, "dr90")
COORDS_PATH = os.path.join(DATA_DIR, "real_stop_coordinates.json")
WINDOW_SIZE = 5

st.set_page_config(page_title="CrowdFlow Dashboard", layout="wide")


def clock_time(minutes_after_midnight):
    """Format a route time such as 390 as 6:30 AM."""
    total_minutes = int(round(minutes_after_midnight)) % (24 * 60)
    hour, minute = divmod(total_minutes, 60)
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {suffix}"


@st.cache_data
def load_real_coordinates():
    """Returns {stop_name: {lat, lng}} if calibration/geocode_stops.py has
    been run, else None -- the map section is skipped gracefully if so."""
    if not os.path.exists(COORDS_PATH):
        return None
    with open(COORDS_PATH) as f:
        data = json.load(f)
    return {name: {"lat": v["lat"], "lng": v["lng"]} for name, v in data.items()}


@st.cache_resource
def load_engine():
    return get_engine()


@st.cache_data
def load_val_trips():
    trip_list = load_trips(os.path.join(SIMULATION_DATA_DIR, "val.csv"))
    trips = {}
    for group in trip_list:
        day = int(group["day"].iloc[0])
        bus_id = int(group["bus_id"].iloc[0])
        trips[(day, bus_id)] = group
    return trips


@st.cache_data(show_spinner="Planning departure scenarios...")
def load_planner_scenario(seed=999):
    """Run one new service-day scenario with the explicit future-path planner."""
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
            "recommended_departure_min": scheduled_departure_min + best["delay_seconds"] / 60.0,
            "closest_stop_name": best["closest_stop_name"],
            "closest_gap_min": best["closest_gap_min"],
            "scheduled_gap_min": best["scheduled_gap_min"],
            "minimum_safe_gap_min": best["minimum_safe_gap_min"],
            "is_safe": best["is_safe"],
            "warning": plan["warning"],
            "candidates": [
                {
                    "Departure delay (s)": candidate["delay_seconds"],
                    "Leave at": clock_time(scheduled_departure_min + candidate["delay_seconds"] / 60.0),
                    "Safe across route": "Yes" if candidate["is_safe"] else "No",
                    "Closest projected stop": candidate["closest_stop_name"],
                    "Closest projected gap (min)": round(candidate["closest_gap_min"], 2),
                    "Minimum safe gap (min)": round(candidate["minimum_safe_gap_min"], 2),
                }
                for candidate in sorted(
                    plan["candidates"],
                    key=lambda item: (item["max_unsafe_ratio"], item["mean_unsafe_ratio"], item["score"]),
                )[:5]
            ],
        }
        _, _, terminated, _, _ = planner.depart_with_delay(best["delay_seconds"])
    return decisions


def get_window(feats, i, window=WINDOW_SIZE):
    start = i - window + 1
    if start < 0:
        pad = np.zeros((abs(start), feats.shape[1]), dtype=np.float32)
        return np.concatenate([pad, feats[0:i + 1]], axis=0)
    return feats[start:i + 1]


@st.cache_data(show_spinner="Finding trips with predicted bunching risk...")
def load_trip_risk_summary():
    """Summarize each trip by its LSTM next-stop risk predictions.

    The dashboard uses this only to make the trip picker useful: a user can
    jump directly to trips that contain Medium or High predictions instead
    of searching the full validation set manually.
    """
    engine = load_engine()
    trip_windows, window_trip_keys = [], []

    for trip_key, trip in load_val_trips().items():
        feats = trip[FEATURE_COLS].to_numpy(dtype=np.float32)
        for i in range(N_STOPS - 3):
            trip_windows.append(get_window(feats, i))
            window_trip_keys.append(trip_key)

    windows = np.asarray(trip_windows, dtype=np.float32)
    windows = (windows - engine.scaler_mean) / engine.scaler_std
    predicted_risks = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[start:start + batch_size], dtype=torch.float32)
            _, logits = engine.lstm(batch)
            predicted_risks.extend(logits.argmax(dim=-1).numpy()[:, 0].tolist())

    summary = {}
    for trip_key, risk_idx in zip(window_trip_keys, predicted_risks):
        if trip_key not in summary:
            summary[trip_key] = {"Low": 0, "Medium": 0, "High": 0}
        summary[trip_key][["Low", "Medium", "High"][risk_idx]] += 1

    return pd.DataFrame([
        {"day": day, "bus_id": bus_id, **counts}
        for (day, bus_id), counts in summary.items()
    ])


st.set_page_config(page_title="CrowdFlow Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #071a2c 0%, #0d2137 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stSidebar"] {
        background: rgba(11, 25, 38, 0.95);
    }
    .stMetric > div {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 0.8rem 1rem;
        border-radius: 0.8rem;
    }
    .metric-label {
        font-size: 0.8rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #a9c3d9;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
    }
    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 1rem;
        padding: 1rem 1.1rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.72rem;
    }
    .badge-low { background: rgba(46, 160, 67, 0.18); color: #7ed995; }
    .badge-medium { background: rgba(230, 180, 30, 0.16); color: #f5d66d; }
    .badge-high { background: rgba(200, 40, 40, 0.18); color: #ff8d8d; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚍 CrowdFlow")
st.caption("MTC Route 21G | Bus bunching monitor and dispatch planner")

engine = load_engine()
trips = load_val_trips()
planner_scenario = load_planner_scenario()
trip_risk_summary = load_trip_risk_summary()

with st.sidebar:
    st.header("Filters")
    trip_filter = st.radio(
        "Trip list",
        options=["All trips", "Trips with Medium or High risk", "Trips with High risk"],
        horizontal=False,
    )

    if trip_filter == "Trips with Medium or High risk":
        visible_summary = trip_risk_summary[(trip_risk_summary["Medium"] + trip_risk_summary["High"]) > 0]
    elif trip_filter == "Trips with High risk":
        visible_summary = trip_risk_summary[trip_risk_summary["High"] > 0]
    else:
        visible_summary = trip_risk_summary

    trip_keys = [
        (int(row.day), int(row.bus_id))
        for row in visible_summary.sort_values(["day", "bus_id"]).itertuples(index=False)
    ]
    risk_counts = {
        (int(row.day), int(row.bus_id)): (int(row.Medium), int(row.High))
        for row in visible_summary.itertuples(index=False)
    }

    selected = st.selectbox(
        "Choose a trip",
        options=trip_keys,
        format_func=lambda k: (
            f"Day {k[0]}, Bus #{k[1]} — "
            f"{risk_counts[k][1]} High / {risk_counts[k][0]} Medium"
        ),
        index=0,
    )

trip = trips[selected]
departure_plan = planner_scenario[selected[1]]
feats = trip[FEATURE_COLS].to_numpy(dtype=np.float32)
stop_names = trip["stop_name"].tolist()
actual_dev = trip["headway_deviation_min"].to_numpy()

rows = []
pred_1step = [np.nan] * N_STOPS
for i in range(0, N_STOPS - 3):
    window = get_window(feats, i)
    window_dicts = [dict(zip(FEATURE_COLS, r)) for r in window]
    pred = engine.predict_bunching(window_dicts)
    pred_1step[i + 1] = pred["predicted_deviation_min"][0]
    risk = pred["risk_class"][0]
    rows.append({
        "Stop": stop_names[i],
        "Actual Deviation (min)": round(float(actual_dev[i]), 2),
        "LSTM Predicted Next-Stop Deviation (min)": round(float(pred["predicted_deviation_min"][0]), 2),
        "Predicted Risk (next stop)": risk,
    })

results_df = pd.DataFrame(rows)
selected_medium, selected_high = risk_counts[selected]

st.markdown("<div class='section-header'>Trip overview</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("High risk stops", int((results_df["Predicted Risk (next stop)"] == "High").sum()))
col2.metric("Medium risk stops", int((results_df["Predicted Risk (next stop)"] == "Medium").sum()))
col3.metric("Scheduled departure", clock_time(departure_plan["scheduled_departure_min"]))
col4.metric("Recommended departure", clock_time(departure_plan["recommended_departure_min"]))

info_col, risk_col = st.columns([2, 1])
with info_col:
    st.markdown(
        f"""
        <div class='glass-card'>
            <div class='section-header'>Selected trip</div>
            <strong>Day {selected[0]}</strong> · Bus #{selected[1]}<br>
            Route: Parrys Corner → Kilambakkam<br>
            Forecast risk summary: {selected_high} high, {selected_medium} medium
        </div>
        """,
        unsafe_allow_html=True,
    )

with risk_col:
    if departure_plan["is_safe"]:
        st.success(f"Safe departure window: {departure_plan['minimum_safe_gap_min']:.1f} min minimum gap")
    else:
        st.warning("Planner could not find a fully safe departure within the default search range")

if departure_plan["warning"]:
    st.warning(departure_plan["warning"])

# Chart + table tabs
chart_tab, details_tab, map_tab = st.tabs(["Forecast trend", "Stop-level details", "Route map"])

with chart_tab:
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="#0d2137")
    ax.plot(range(N_STOPS), actual_dev, marker='o', label='Actual deviation', color='#ff6b6b', linewidth=2.3)
    ax.plot(range(N_STOPS), pred_1step, marker='x', linestyle='--', label='LSTM 1-step prediction', color='#5fb3ff', linewidth=2)
    ax.axhline(0, color='white', linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Stop index", color='white')
    ax.set_ylabel("Headway deviation (min)", color='white')
    ax.tick_params(colors='white')
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=9)
    for spine in ax.spines.values():
        spine.set_color('#b7d1eb')
    fig.tight_layout()
    st.pyplot(fig)

with details_tab:
    def highlight_risk(row):
        color = {
            "Low": "background-color: rgba(46, 160, 67, 0.18); color: #d9fdd6",
            "Medium": "background-color: rgba(230, 180, 30, 0.18); color: #ffe9a6",
            "High": "background-color: rgba(200, 40, 40, 0.18); color: #ffd8d8",
        }[row["Predicted Risk (next stop)"]]
        return [color] * len(row)

    st.dataframe(
        results_df.style.apply(highlight_risk, axis=1),
        use_container_width=True,
        height=420,
        hide_index=True,
    )

with map_tab:
    real_coords = load_real_coordinates()
    if real_coords is None:
        st.info(
            "No real stop coordinates found. Run the geocoding calibration to enable the live route map."
        )
    else:
        RISK_COLOR = {"Low": [46, 160, 67], "Medium": [230, 180, 30], "High": [200, 40, 40]}
        map_rows = []
        for row in rows:
            name = row["Stop"]
            if name not in real_coords:
                continue
            map_rows.append({
                "name": name,
                "lat": real_coords[name]["lat"],
                "lon": real_coords[name]["lng"],
                "risk": row["Predicted Risk (next stop)"],
                "color": RISK_COLOR[row["Predicted Risk (next stop)"]],
            })
        map_df = pd.DataFrame(map_rows)
        path_coords = [[real_coords[n]["lng"], real_coords[n]["lat"]] for n in stop_names if n in real_coords]

        if map_df.empty:
            st.warning("No mapped stop coordinates matched the selected trip route.")
        else:
            layers = [
                pdk.Layer("PathLayer", data=[{"path": path_coords}], get_path="path", get_color=[130, 130, 130], width_min_pixels=3),
                pdk.Layer("ScatterplotLayer", data=map_df, get_position=["lon", "lat"], get_fill_color="color", get_radius=120, pickable=True),
            ]
            view_state = pdk.ViewState(latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(), zoom=10.2)
            st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{name}\nRisk: {risk}"}, map_style="road"))

st.markdown("<div class='section-header'>Departure recommendation</div>", unsafe_allow_html=True)

planner_cols = st.columns(4)
planner_cols[0].metric("Scheduled departure", clock_time(departure_plan["scheduled_departure_min"]))
planner_cols[1].metric("Recommended delay", f"{departure_plan['recommended_delay_seconds']} sec")
planner_cols[2].metric("Recommended departure", clock_time(departure_plan["recommended_departure_min"]))
planner_cols[3].metric("Closest projected gap", f"{departure_plan['closest_gap_min']:.1f} min")

if departure_plan["recommended_delay_seconds"] == 0:
    st.caption("Decision: depart now — the scheduled departure is already safe.")
else:
    st.caption(f"Recommended wait: {departure_plan['recommended_delay_seconds']} seconds after the scheduled departure.")

if departure_plan["warning"]:
    st.warning(departure_plan["warning"])
elif departure_plan["is_safe"]:
    st.success(
        f"This start time keeps the next bus at least {departure_plan['minimum_safe_gap_min']:.1f} minutes "
        "behind the previous bus across the projected route."
    )

with st.expander("Counterfactual departure candidates"):
    st.dataframe(pd.DataFrame(departure_plan["candidates"]), use_container_width=True, hide_index=True)

st.caption(
    "This dashboard compares actual trip behavior with LSTM next-stop predictions and a fresh departure-planning simulation."
)
