"""
app.py — Monitor & Alert Dashboard
-----------------------------------
Solar-Powered Smart Mini Cold Storage — a single unit (Cooling Station),
one shared solar/battery system subdivided into independent compartments
(one per vegetable type).

Run with:
    streamlit run app.py

Requires spoilage_model.pkl (run `python train_model.py` first if missing).
"""

import streamlit as st
import pandas as pd
import joblib

from simulator import (
    default_fleet, collect_alerts, unit_health, compartment_health, CROPS,
)

st.set_page_config(page_title="Cold Storage Dashboard", layout="wide")

HEALTH_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
CONNECTIVITY_EMOJI = {"wifi": "📶", "gsm": "📱"}
RISK_EMOJI = {"Safe": "🟢", "Moderate": "🟡", "Act Now": "🔴"}

# ---------------- Model ----------------
@st.cache_resource
def load_model():
    bundle = joblib.load("spoilage_model.pkl")
    return bundle["model"], bundle["feature_cols"], bundle["crops"]

model, feature_cols, crops = load_model()


def predict_risk(reading):
    """AI spoilage-risk prediction for Auto-mode compartments; simple
    deviation-from-target status for Manual-mode compartments (no
    crop-specific shelf-life data exists for an arbitrary manual target)."""
    if reading["mode"] != "auto":
        dev = abs(reading["internal_temp_c"] - reading["target_temp"])
        if dev > 4:
            return "Act Now"
        elif dev > 2:
            return "Moderate"
        return "Safe"

    row = {
        "internal_temp_c": reading["internal_temp_c"],
        "internal_humidity_pct": reading["internal_humidity_pct"],
        "hours_since_harvest": reading["harvest_hours"],
        "cooling_uptime_pct": reading["cooling_uptime_pct"],
    }
    for c in crops:
        row[f"crop_{c}"] = 1 if c == reading["crop"] else 0
    X = pd.DataFrame([row])
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]
    return model.predict(X)[0]


# ---------------- Session state ----------------
if "fleet" not in st.session_state:
    st.session_state.fleet = default_fleet()
    st.session_state.history = []  # list of {unit_id, compartment_id, total_hours, battery_pct, solar_output_w, **reading}

fleet = st.session_state.fleet
unit_id = next(iter(fleet.units))  # single unit
unit = fleet.units[unit_id]

# ---------------- Sidebar: global controls ----------------
st.sidebar.title("⚙️ Simulation Controls")
step_minutes = st.sidebar.slider("Advance time (minutes)", 15, 120, 60, step=15)
col_a, col_b = st.sidebar.columns(2)
advance = col_a.button("▶ Advance")
reset = col_b.button("⟲ Reset")
cloudy = st.sidebar.toggle("☁️ Simulate cloudy day", value=False)

low_bandwidth = st.sidebar.toggle("📶 Low-connectivity / icon mode", value=False)
st.sidebar.caption(
    "Icon-heavy, chart-free view for low-bandwidth GSM/SMS sync conditions."
)

if reset:
    st.session_state.fleet = default_fleet()
    st.session_state.history = []
    st.rerun()

if advance:
    fleet_readings = fleet.step(minutes=step_minutes, cloudy=cloudy)
    for uid, (unit_reading, compartment_readings) in fleet_readings.items():
        for cid, reading in compartment_readings.items():
            entry = {
                "unit_id": uid, "compartment_id": cid,
                "total_hours": unit_reading["total_hours"],
                "sim_time_hours": unit_reading["sim_time_hours"],
                "battery_pct": unit_reading["battery_pct"],
                "solar_output_w": unit_reading["solar_output_w"],
                **reading,
            }
            st.session_state.history.append(entry)
else:
    # Zero-duration snapshot: reflects any control changes (mode, door
    # toggle) immediately without advancing simulated time or logging
    # a new history point.
    fleet_readings = fleet.step(minutes=0, cloudy=cloudy)

unit_reading, compartment_readings = fleet_readings[unit_id]

# ---------------- Header ----------------
st.title("🧊 Cold Storage — Monitor & Alert Dashboard")

# ============================================================
# Helper renderers
# ============================================================

def render_compartment_card_full(cid, reading, unit_battery_pct, unit_solar_w):
    level, issues = compartment_health(reading, unit_battery_pct, unit_solar_w)
    risk = predict_risk(reading)

    with st.container(border=True):
        top = st.columns([3, 1])
        top[0].markdown(f"**{reading['compartment_label']}** {HEALTH_EMOJI[level]}")
        top[1].markdown(f"{RISK_EMOJI[risk]} {risk}")

        # Two rows of metrics so values never get cramped/truncated
        row1 = st.columns(3)
        row1[0].metric("Temp", f"{reading['internal_temp_c']}°C")
        row1[1].metric("Humidity", f"{reading['internal_humidity_pct']}%")
        row1[2].metric("Door", "🚪 Open" if reading["door_open"] else "✅ Closed")

        row2 = st.columns(2)
        row2[0].metric("Battery", f"{unit_battery_pct}%")
        row2[1].metric("Solar", f"{unit_solar_w} W")
        st.caption("Battery & solar are shared across this unit's compartments.")

        mode_label = (
            f"Auto — {reading['crop'].replace('_',' ').title()}"
            if reading["mode"] == "auto" else f"Manual — target {reading['manual_target']}°C"
        )
        st.caption(f"Mode: {mode_label}  |  Target: {reading['target_temp']}°C  |  Cooling uptime: {reading['cooling_uptime_pct']}%")

        if issues:
            st.warning(", ".join(text for text, _ in issues))

        with st.expander("⚙️ Control this compartment"):
            new_mode = st.radio(
                "Mode", ["Auto (by crop)", "Manual (fixed target)"],
                index=0 if reading["mode"] == "auto" else 1,
                key=f"mode_radio_{cid}",
            )
            if new_mode.startswith("Auto"):
                crop_choice = st.selectbox(
                    "Crop", list(CROPS.keys()),
                    index=list(CROPS.keys()).index(reading["crop"]) if reading["mode"] == "auto" else 0,
                    key=f"crop_select_{cid}",
                    format_func=lambda c: c.replace("_", " ").title(),
                )
                if st.button("Apply", key=f"apply_auto_{cid}"):
                    unit.compartments[cid].set_mode_auto(crop_choice)
                    st.rerun()
            else:
                target = st.number_input(
                    "Target temperature (°C)", min_value=0.0, max_value=20.0,
                    value=float(reading["manual_target"] or 6.0), step=0.5,
                    key=f"target_input_{cid}",
                )
                if st.button("Apply", key=f"apply_manual_{cid}"):
                    unit.compartments[cid].set_mode_manual(target)
                    st.rerun()
            if st.button("🚪 Toggle door (demo)", key=f"door_{cid}"):
                unit.compartments[cid].force_door(not reading["door_open"], unit.time_hours)
                st.rerun()


def render_compartment_card_icon(reading, unit_battery_pct, unit_solar_w):
    level, issues = compartment_health(reading, unit_battery_pct, unit_solar_w)
    risk = predict_risk(reading)
    door_icon = "🚪" if reading["door_open"] else "✅"
    batt_icon = "🪫" if unit_battery_pct < 35 else "🔋"
    st.markdown(
        f"{HEALTH_EMOJI[level]} **{reading['compartment_label']}**  "
        f"🌡️{reading['internal_temp_c']}°C  "
        f"💧{reading['internal_humidity_pct']}%  "
        f"{batt_icon}{unit_battery_pct}%  "
        f"☀️{unit_solar_w}W  "
        f"{door_icon}  "
        f"{RISK_EMOJI[risk]}{risk}"
    )


def render_unit_header():
    level, _ = unit_health(unit_reading, compartment_readings)
    sync_info = (
        "Synced live"
        if unit.connectivity == "wifi"
        else f"Last synced {int(unit.last_synced_minutes_ago)} min ago via SMS"
    )
    st.subheader(f"{HEALTH_EMOJI[level]} {unit.name} — {unit.location}")
    st.caption(f"{CONNECTIVITY_EMOJI[unit.connectivity]} {unit.connectivity.upper()}  •  {sync_info}")
    m = st.columns(3)
    m[0].metric("🔋 Shared Battery", f"{unit_reading['battery_pct']}%")
    m[1].metric("☀️ Solar Input", f"{unit_reading['solar_output_w']} W")
    m[2].metric("🌡️ Ambient", f"{unit_reading['ambient_temp_c']}°C")
    st.caption("Battery and solar are shared across all compartments — running more coolers at once drains the battery faster.")


def _resample(df, granularity):
    if granularity == "Daily average":
        df = df.groupby(df.index // 24).mean()
        df.index.name = "day"
    elif granularity == "Weekly average":
        df = df.groupby(df.index // (24 * 7)).mean()
        df.index.name = "week"
    return df


def render_compartment_trend_chart(cid):
    """Temperature & humidity trend — specific to this compartment."""
    rows = [h for h in st.session_state.history if h["compartment_id"] == cid]
    if len(rows) < 2:
        st.caption("Not enough history yet — advance time to build a trend.")
        return
    df = pd.DataFrame(rows)
    granularity = st.radio(
        "Granularity", ["Hourly (raw)", "Daily average", "Weekly average"],
        horizontal=True, key=f"gran_{cid}",
    )
    df = df.set_index("total_hours")[["internal_temp_c", "internal_humidity_pct"]]
    st.line_chart(_resample(df, granularity))


def render_power_trend_chart():
    """Battery % & solar output trend — shared at the unit level, so
    sourced once rather than duplicated per compartment."""
    first_cid = next(iter(unit.compartments))
    rows = [h for h in st.session_state.history if h["compartment_id"] == first_cid]
    if len(rows) < 2:
        st.caption("Not enough history yet — advance time to build a trend.")
        return
    df = pd.DataFrame(rows)
    granularity = st.radio(
        "Granularity", ["Hourly (raw)", "Daily average", "Weekly average"],
        horizontal=True, key="gran_power",
    )
    df = df.set_index("total_hours")[["battery_pct", "solar_output_w"]]
    st.line_chart(_resample(df, granularity))


# ============================================================
# Main layout (single unit — no fleet-level view needed)
# ============================================================

render_unit_header()

st.markdown("#### Compartments")
if low_bandwidth:
    for cid, reading in compartment_readings.items():
        render_compartment_card_icon(reading, unit_reading["battery_pct"], unit_reading["solar_output_w"])
else:
    cols = st.columns(len(compartment_readings))
    for i, (cid, reading) in enumerate(compartment_readings.items()):
        with cols[i]:
            render_compartment_card_full(cid, reading, unit_reading["battery_pct"], unit_reading["solar_output_w"])

st.markdown("---")
st.markdown("#### 🔔 Alerts")
alerts = collect_alerts(fleet_readings, fleet.units)
if alerts:
    for a in sorted(alerts, key=lambda x: {"red": 0, "yellow": 1}.get(x["level"], 2)):
        st.markdown(f"{HEALTH_EMOJI[a['level']]} **[{a['compartment']}]** {a['issue']}")
else:
    st.success("No active alerts.")

if not low_bandwidth:
    st.markdown("---")
    st.markdown("#### 📈 Historical Trends")

    st.markdown("**⚡ Power (shared across compartments)**")
    render_power_trend_chart()

    st.markdown("**🌡️ Temperature & Humidity (per compartment)**")
    comp_tabs = st.tabs([r["compartment_label"] for r in compartment_readings.values()])
    for tab, (cid, reading) in zip(comp_tabs, compartment_readings.items()):
        with tab:
            render_compartment_trend_chart(cid)
else:
    st.caption("📉 Trend charts are hidden in low-connectivity mode to save bandwidth.")
