# Cold Storage Fleet — Monitor & Alert Dashboard (v2)

Models a **cooperative fleet of cold storage units**. Each unit is a single
physical zone with **one shared solar panel + battery**, subdivided into
independent **compartments** — one per vegetable type.

## Data model

```
Fleet
 └─ Unit (one physical installation = one zone, one shared solar+battery)
     ├─ Compartment (e.g. "Tomato Compartment") — own crop/target, temp, door
     └─ Compartment (e.g. "Leafy Greens Compartment") — own crop/target, temp, door
```

Compartments run independent cooling elements but **draw from the same
battery**, so running two compartments' coolers at once drains the shared
battery noticeably faster than running one — a realistic trade-off worth
mentioning in a pitch.

## Features

- **Live data per compartment** — temperature, humidity, door status, mode.
  Battery % and solar input are shown once per unit (shared), not duplicated
  per compartment.
- **Auto vs Manual mode per compartment**
  - *Auto*: targets a selected crop's ideal storage temperature; spoilage
    risk is predicted by the trained ML model (Safe / Moderate / Act Now).
  - *Manual*: targets an operator-set fixed temperature; status is judged
    by simple deviation-from-target thresholds (no crop-specific shelf-life
    data exists for an arbitrary manual target).
- **Door status** — each compartment simulates occasional door-open events,
  with duration tracked and flagged if left open too long.
- **Alerts feed** — tagged by unit + compartment, each issue carries its own
  severity (temperature excursion, low battery, door left open, power
  failure, low cooling uptime).
- **Historical trends** — per-compartment temperature/humidity charts,
  viewable hourly (raw), daily average, or weekly average.
- **Multi-unit fleet view** — a cooperative-level overview showing
  🟢/🟡/🔴 health status across all deployed units at a glance.
- **Role-based access**
  - *Farmer*: read-only view of their own assigned unit.
  - *Cooperative Manager/Admin*: full fleet view, can switch any
    compartment between Auto/Manual, change crop or target temperature,
    and force a door-open demo event.
- **Low-connectivity mode** — a toggle that switches to an icon-heavy,
  chart-free layout, and each unit shows its sync method (📶 WiFi, live;
  or 📱 GSM/SMS, with a simulated periodic sync lag) — reflecting real NER
  deployment conditions where WiFi isn't always available.

## Demo fleet (pre-configured)

| Unit | Location | Connectivity | Compartments |
|---|---|---|---|
| AgriNex Cold Hub | Sangamner Farm Cluster | WiFi | Tomato (Auto), Leafy Greens (Auto) |
| Cooling Station | Kohima Village Market | GSM/SMS | Custom (Manual, 6°C), Chili (Auto) |
| Village Node | Imphal Farm Gate | GSM/SMS | Cucumber (Auto) |

## How to run

```bash
pip install -r requirements.txt
python train_model.py      # only needed if spoilage_model.pkl is missing
streamlit run app.py
```

## Suggested demo flow

1. Log in as **Cooperative Manager/Admin** → show the **Fleet Overview**:
   three units, color-coded health, connectivity badges, shared battery %.
2. Drill into a unit → show its compartments, each with live readings,
   mode, and AI/deviation-based risk — note battery/solar shown once per
   unit, not per compartment.
3. Open a compartment's **Control** panel → switch it from Auto to Manual
   (or change its crop) → show the card update immediately.
4. Click **"Toggle door (demo)"** on a compartment → show the door-open
   alert appear in the Alerts feed.
5. Toggle **"Simulate cloudy day"** and advance time a few times → show
   the shared battery drain faster when multiple compartments are cooling
   at once, cooling uptime drop, and risk climb.
6. Switch to **Farmer** role → show the scoped-down, read-only single-unit
   view.
7. Toggle **Low-connectivity / icon mode** → show the simplified,
   chart-free layout designed for GSM/SMS-only conditions.
8. Check the **Historical Trends** tab on a compartment → switch between
   hourly, daily, and weekly views.

## Notes

- All data is simulated — no physical hardware or real GSM modem required.
  The GSM "sync lag" and WiFi "live sync" are both simulated for demo
  purposes.
- The spoilage-risk model is unchanged from the original prototype — same
  synthetic, physics-informed training data grounded in published
  vegetable shelf-life tables.
