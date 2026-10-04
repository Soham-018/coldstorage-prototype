"""
train_model.py
----------------
Generates a synthetic, physics-informed dataset for vegetable spoilage risk
based on published ideal storage conditions, then trains a RandomForest
classifier to predict risk category (Safe / Moderate / Act Now) from
live storage conditions.

Why synthetic data: no real IoT deployment data exists yet for this NER
use case. We ground the simulation in known ideal storage temp/humidity
and shelf-life ranges for common NER vegetables, and model degradation
as accelerating exponentially the further conditions drift from ideal.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

np.random.seed(42)

# Ideal storage conditions and baseline shelf life (days) for common
# NER vegetables, approximated from agricultural extension references.
CROPS = {
    "tomato":        {"ideal_temp": 13, "ideal_humidity": 90, "base_shelf_life": 14},
    "leafy_greens":  {"ideal_temp": 2,  "ideal_humidity": 95, "base_shelf_life": 10},
    "chili":         {"ideal_temp": 10, "ideal_humidity": 90, "base_shelf_life": 12},
    "cucumber":      {"ideal_temp": 12, "ideal_humidity": 90, "base_shelf_life": 10},
    "cabbage":       {"ideal_temp": 3,  "ideal_humidity": 95, "base_shelf_life": 60},
}

RISK_LABELS = ["Safe", "Moderate", "Act Now"]


def simulate_row():
    crop = np.random.choice(list(CROPS.keys()))
    ideal = CROPS[crop]

    # Simulate actual storage conditions with realistic deviation
    temp = np.random.normal(ideal["ideal_temp"], 4)
    humidity = np.clip(np.random.normal(ideal["ideal_humidity"], 8), 30, 100)
    hours_since_harvest = np.random.uniform(0, ideal["base_shelf_life"] * 24)
    cooling_uptime_pct = np.clip(np.random.normal(80, 20), 0, 100)  # % time compressor kept temp in range

    # Degradation factor: how far conditions deviate from ideal
    temp_dev = abs(temp - ideal["ideal_temp"])
    humidity_dev = abs(humidity - ideal["ideal_humidity"])

    # Effective shelf life shrinks exponentially with deviation and
    # inconsistent cooling (physics-informed heuristic, not measured data)
    decay_factor = np.exp(-0.08 * temp_dev - 0.02 * humidity_dev) * (cooling_uptime_pct / 100)
    effective_shelf_life_hours = ideal["base_shelf_life"] * 24 * decay_factor

    days_remaining = max(0, (effective_shelf_life_hours - hours_since_harvest) / 24)

    if days_remaining > 3:
        risk = "Safe"
    elif days_remaining > 1:
        risk = "Moderate"
    else:
        risk = "Act Now"

    return {
        "crop": crop,
        "internal_temp_c": round(temp, 1),
        "internal_humidity_pct": round(humidity, 1),
        "hours_since_harvest": round(hours_since_harvest, 1),
        "cooling_uptime_pct": round(cooling_uptime_pct, 1),
        "days_remaining": round(days_remaining, 2),
        "risk": risk,
    }


def generate_dataset(n=4000):
    rows = [simulate_row() for _ in range(n)]
    df = pd.DataFrame(rows)
    return df


def train():
    df = generate_dataset()
    df.to_csv("synthetic_storage_data.csv", index=False)
    print(f"Generated {len(df)} synthetic samples -> synthetic_storage_data.csv")
    print(df["risk"].value_counts())

    df_enc = pd.get_dummies(df, columns=["crop"])
    feature_cols = [c for c in df_enc.columns if c not in ("risk", "days_remaining")]

    X = df_enc[feature_cols]
    y = df_enc["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nModel evaluation on held-out test set:")
    print(classification_report(y_test, preds))

    joblib.dump({"model": model, "feature_cols": feature_cols, "crops": list(CROPS.keys())},
                "spoilage_model.pkl")
    print("Saved trained model -> spoilage_model.pkl")


if __name__ == "__main__":
    train()
