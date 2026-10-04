"""
simulator.py
------------
Multi-compartment, multi-unit solar cold storage simulator.

Hierarchy:
    Fleet -> Unit (one physical cold storage installation = ONE zone,
             with ONE shared solar panel + battery + power system)
             -> Compartment (an independently climate-controlled section
             inside that zone, e.g. one per vegetable type).

Compartments share the unit's single battery/solar system but each runs
its own cooling element, so each can target a different crop or manual
temperature independently. Running more compartments' coolers at once
draws down the shared battery faster — a realistic trade-off.

Each Compartment can run in:
    - "auto"   mode: targets the ideal storage temp for a selected crop,
                     and spoilage risk is predicted by the trained ML model.
    - "manual" mode: targets a fixed temperature set by the operator,
                     and status is judged by simple deviation-from-target
                     rules (no crop-specific shelf-life data available).

Connectivity per unit is either "wifi" (near-instant sync) or
"gsm" (periodic SMS-style sync with a simulated lag), to reflect
real NER deployment conditions.
"""

import numpy as np

CROPS = {
    "tomato":       {"ideal_temp": 13, "ideal_humidity": 90, "base_shelf_life": 14},
    "leafy_greens": {"ideal_temp": 2,  "ideal_humidity": 95, "base_shelf_life": 10},
    "chili":        {"ideal_temp": 10, "ideal_humidity": 90, "base_shelf_life": 12},
    "cucumber":     {"ideal_temp": 12, "ideal_humidity": 90, "base_shelf_life": 10},
    "cabbage":      {"ideal_temp": 3,  "ideal_humidity": 95, "base_shelf_life": 60},
}


class CompartmentSimulator:
    """Simulates one climate-controlled compartment within a unit's single
    zone. Has its own crop/target, temperature, humidity, and door — but
    NOT its own battery or solar panel (those are shared at the Unit level)."""

    COMPRESSOR_DRAIN_PCT_PER_HOUR = 15.0  # draw while THIS compartment's cooler runs

    def __init__(self, compartment_id, label, mode="auto", crop="tomato", manual_target=6.0):
        self.compartment_id = compartment_id
        self.label = label
        self.mode = mode                  # "auto" | "manual"
        self.crop = crop
        self.manual_target = manual_target
        self.internal_temp = 12.0
        self.internal_humidity = 88.0
        self.door_open = False
        self.door_open_since_hours = None
        self.cooling_uptime_history = []
        self.harvest_hours = 6.0

    def set_mode_auto(self, crop):
        self.mode = "auto"
        self.crop = crop
        self.harvest_hours = 0.0  # fresh batch assumed on crop change

    def set_mode_manual(self, target_temp):
        self.mode = "manual"
        self.manual_target = target_temp

    def target_temp(self):
        if self.mode == "manual":
            return self.manual_target
        return CROPS[self.crop]["ideal_temp"]

    def force_door(self, open_: bool, now_hours):
        self.door_open = open_
        self.door_open_since_hours = now_hours if open_ else None

    def _maybe_toggle_door(self, now_hours, p_open=0.03, p_close=0.5):
        if self.door_open:
            if np.random.random() < p_close:
                self.door_open = False
                self.door_open_since_hours = None
        else:
            if np.random.random() < p_open:
                self.door_open = True
                self.door_open_since_hours = now_hours

    def step(self, hours_elapsed, now_hours, ambient_temp, battery_available):
        """Advances this compartment's climate. Returns (reading, battery_draw_pct)
        so the parent Unit can apply the shared drain."""
        is_snapshot = hours_elapsed == 0

        if not is_snapshot:
            self._maybe_toggle_door(now_hours)

        target = self.target_temp()
        must_cool = self.internal_temp > target + 2
        compressor_on = must_cool and battery_available

        draw_pct = 0.0
        if not is_snapshot:
            if compressor_on:
                self.internal_temp -= 1.2
                draw_pct = self.COMPRESSOR_DRAIN_PCT_PER_HOUR * hours_elapsed
            else:
                self.internal_temp += 0.4

            if self.door_open:
                self.internal_temp += 0.6
                self.internal_humidity -= 1.0

            self.internal_temp = np.clip(self.internal_temp, 0, ambient_temp)
            self.internal_humidity = np.clip(self.internal_humidity + np.random.normal(0, 1), 50, 100)

            self.cooling_uptime_history.append(1 if compressor_on else 0)
            if len(self.cooling_uptime_history) > 96:
                self.cooling_uptime_history.pop(0)

        cooling_uptime_pct = 100 * np.mean(self.cooling_uptime_history) if self.cooling_uptime_history else 100

        door_open_minutes = 0.0
        if self.door_open and self.door_open_since_hours is not None:
            door_open_minutes = (now_hours - self.door_open_since_hours) * 60

        reading = {
            "compartment_id": self.compartment_id,
            "compartment_label": self.label,
            "mode": self.mode,
            "crop": self.crop if self.mode == "auto" else None,
            "manual_target": round(self.manual_target, 1) if self.mode == "manual" else None,
            "target_temp": round(target, 1),
            "internal_temp_c": round(self.internal_temp, 1),
            "internal_humidity_pct": round(self.internal_humidity, 1),
            "compressor_on": compressor_on,
            "cooling_uptime_pct": round(cooling_uptime_pct, 1),
            "door_open": self.door_open,
            "door_open_minutes": round(door_open_minutes, 1),
            "harvest_hours": round(self.harvest_hours, 1),
        }
        return reading, draw_pct


class Unit:
    """One physical cold storage installation: a single zone with one
    shared solar panel + battery, subdivided into independent compartments
    (e.g. one per vegetable type)."""

    SOLAR_TO_BATTERY_PCT_PER_WATT_HOUR = 0.06
    BASELINE_DRAIN_PCT_PER_HOUR = 2.0  # shared electronics/sensors, always on

    def __init__(self, unit_id, name, location, compartments_config,
                 connectivity="wifi", start_hour=6.0, battery_pct=70.0):
        self.unit_id = unit_id
        self.name = name
        self.location = location
        self.connectivity = connectivity  # "wifi" | "gsm"
        self.time_hours = start_hour
        self.battery_pct = battery_pct
        self.cloudy_override = False
        self.last_synced_minutes_ago = 0.0
        self.compartments = {
            c["compartment_id"]: CompartmentSimulator(
                compartment_id=c["compartment_id"], label=c["label"],
                mode=c.get("mode", "auto"), crop=c.get("crop", "tomato"),
                manual_target=c.get("manual_target", 6.0),
            )
            for c in compartments_config
        }

    def set_cloudy(self, is_cloudy: bool):
        self.cloudy_override = is_cloudy

    def _solar_output_w(self):
        hour = self.time_hours % 24
        base = max(0.0, np.sin((hour - 6) / 12 * np.pi)) * 300
        if self.cloudy_override:
            base *= 0.15
        return max(0.0, base + np.random.normal(0, 10))

    def _ambient_conditions(self):
        hour = self.time_hours % 24
        temp = 24 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi)
        humidity = np.clip(75 + np.random.normal(0, 3), 40, 100)
        return temp, humidity

    def step(self, minutes=15, cloudy=False):
        self.set_cloudy(cloudy)
        hours_elapsed = minutes / 60
        is_snapshot = hours_elapsed == 0

        if not is_snapshot:
            self.time_hours += hours_elapsed
            for comp in self.compartments.values():
                comp.harvest_hours += hours_elapsed

        solar_w = self._solar_output_w() if not is_snapshot else 0.0
        ambient_temp, ambient_humidity = self._ambient_conditions()
        battery_available = self.battery_pct > 10

        compartment_readings = {}
        total_draw_pct = 0.0
        for cid, comp in self.compartments.items():
            reading, draw_pct = comp.step(hours_elapsed, self.time_hours, ambient_temp, battery_available)
            compartment_readings[cid] = reading
            total_draw_pct += draw_pct

        if not is_snapshot:
            charge_pct = solar_w * self.SOLAR_TO_BATTERY_PCT_PER_WATT_HOUR * hours_elapsed
            drain_pct = self.BASELINE_DRAIN_PCT_PER_HOUR * hours_elapsed + total_draw_pct
            self.battery_pct = np.clip(self.battery_pct + charge_pct - drain_pct, 0, 100)

            if self.connectivity == "gsm":
                self.last_synced_minutes_ago = float(np.random.uniform(15, 90))
            else:
                self.last_synced_minutes_ago = 0.0

        unit_reading = {
            "sim_time_hours": round(self.time_hours % 24, 2),
            "total_hours": round(self.time_hours, 2),
            "solar_output_w": round(solar_w, 1),
            "battery_pct": round(self.battery_pct, 1),
            "ambient_temp_c": round(ambient_temp, 1),
            "ambient_humidity_pct": round(ambient_humidity, 1),
        }
        return unit_reading, compartment_readings


class Fleet:
    """A cooperative's collection of deployed units."""

    def __init__(self, units):
        self.units = {u.unit_id: u for u in units}

    def step(self, minutes=15, cloudy=False):
        """Returns {unit_id: (unit_reading, {compartment_id: reading})}"""
        out = {}
        for uid, unit in self.units.items():
            out[uid] = unit.step(minutes=minutes, cloudy=cloudy)
        return out


def default_fleet():
    """Builds the demo fleet used by the dashboard: a single unit
    (Cooling Station) with two generically-labeled compartments."""
    unit_b = Unit(
        unit_id="unit_b",
        name="Cooling Station",
        location="Kohima Village Market",
        connectivity="gsm",
        compartments_config=[
            {"compartment_id": "b1", "label": "Compartment 1", "mode": "manual", "manual_target": 6.0},
            {"compartment_id": "b2", "label": "Compartment 2", "mode": "auto", "crop": "chili"},
        ],
    )
    return Fleet([unit_b])


# ---------- health / alert evaluation ----------

_LEVEL_ORDER = {"green": 0, "yellow": 1, "red": 2}


def compartment_health(reading, unit_battery_pct, unit_solar_w):
    """Returns ('green'|'yellow'|'red', [(issue_text, issue_level), ...])
    for one compartment, factoring in the unit's shared battery/solar."""
    issues = []

    temp_dev = abs(reading["internal_temp_c"] - reading["target_temp"])

    if unit_battery_pct < 15:
        issues.append(("Low battery", "red"))
    elif unit_battery_pct < 35:
        issues.append(("Battery below 35%", "yellow"))

    if reading["door_open"] and reading["door_open_minutes"] > 10:
        issues.append(("Door left open", "red"))
    elif reading["door_open"]:
        issues.append(("Door open", "yellow"))

    if temp_dev > 4:
        issues.append(("Temperature excursion", "red"))
    elif temp_dev > 2:
        issues.append(("Temperature drifting", "yellow"))

    if unit_battery_pct <= 0 and unit_solar_w < 5:
        issues.append(("Power failure", "red"))

    if reading["cooling_uptime_pct"] < 50:
        issues.append(("Low cooling uptime", "yellow"))

    overall = "green"
    for _, lvl in issues:
        if _LEVEL_ORDER[lvl] > _LEVEL_ORDER[overall]:
            overall = lvl

    return overall, issues


def unit_health(unit_reading, compartment_readings):
    """Worst-case health level across a unit's compartments, plus all
    (compartment_label, issue, level) tuples."""
    worst = "green"
    all_issues = []
    for cid, reading in compartment_readings.items():
        level, issues = compartment_health(reading, unit_reading["battery_pct"], unit_reading["solar_output_w"])
        if _LEVEL_ORDER[level] > _LEVEL_ORDER[worst]:
            worst = level
        for issue_text, issue_level in issues:
            all_issues.append((reading["compartment_label"], issue_text, issue_level))
    return worst, all_issues


def collect_alerts(fleet_readings, units_meta):
    """Builds a flat, tagged alert feed across the whole fleet, each issue
    carrying its own severity level.

    fleet_readings: {unit_id: (unit_reading, {compartment_id: reading})}
    """
    alerts = []
    for uid, (unit_reading, compartment_readings) in fleet_readings.items():
        unit_name = units_meta[uid].name
        for cid, reading in compartment_readings.items():
            _, issues = compartment_health(reading, unit_reading["battery_pct"], unit_reading["solar_output_w"])
            for issue_text, issue_level in issues:
                alerts.append({
                    "unit": unit_name,
                    "unit_id": uid,
                    "compartment": reading["compartment_label"],
                    "compartment_id": cid,
                    "issue": issue_text,
                    "level": issue_level,
                })
    return alerts
