# ============================================================
# CNN + BiLSTM Electricity Usage Prediction (With Carbon Footprint) - Final Fixed Version
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import matplotlib.pyplot as plt

# ------------------- User Inputs -------------------
mode = input("Prediction mode (whole_day / specific_hour): ").strip().lower()
date = input("Enter date (YYYY-MM-DD): ").strip()
room_id = input("Enter room ID (e.g., C101, C102, L201, L202): ").strip()
event_flag = int(input("Enter event flag (0/1): ").strip())
exam_flag = int(input("Enter exam flag (0/1): ").strip())
occupancy = int(input("Enter occupancy (number of people): ").strip())

if mode == "specific_hour":
    hour = int(input("Enter hour (0–23): ").strip())

# ------------------- Load Saved Model and Scalers -------------------
base_path = os.path.join(os.getcwd(), "saved_cnn_bilstm_model")

model_path = os.path.join(base_path, "cnn_bilstm_electricity_model.h5")
scaler_X_path = os.path.join(base_path, "scaler_X.pkl")
scaler_y_path = os.path.join(base_path, "scaler_y.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at: {model_path}")
if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
    raise FileNotFoundError("❌ Scalers not found. Please ensure training is complete.")

model = load_model(model_path, compile=False)
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

print("✅ CNN + BiLSTM model and scalers loaded successfully!\n")

# ------------------- Room Info -------------------
room_info = {
    "C101": {"area_m2": 50, "room_type_lab": 0},
    "C102": {"area_m2": 60, "room_type_lab": 0},
    "L201": {"area_m2": 70, "room_type_lab": 1},
    "L202": {"area_m2": 80, "room_type_lab": 1},
}

if room_id not in room_info:
    raise ValueError(f"❌ Invalid room ID! Choose from {list(room_info.keys())}")

# ------------------- Feature Generator -------------------
def generate_features(hour, occupancy):
    dayofweek = pd.to_datetime(date).weekday()
    is_weekend = int(dayofweek >= 5)
    day_of_year = pd.to_datetime(date).dayofyear

    # Approximate environmental features
    outdoor_temp_C = 20 + 10 * np.sin((day_of_year / 365) * 2 * np.pi)
    solar_irradiance_Wm2 = max(
        0, 800 * np.sin((hour / 24) * np.pi) * np.sin((day_of_year / 365) * np.pi)
    )

    # Cyclic encodings
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)
    sin_dayofweek = np.sin(2 * np.pi * dayofweek / 7)
    cos_dayofweek = np.cos(2 * np.pi * dayofweek / 7)

    return {
        "hour": hour,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend,
        "occupancy": occupancy,
        "outdoor_temp_C": outdoor_temp_C,
        "solar_irradiance_Wm2": solar_irradiance_Wm2,
        "exam_flag": exam_flag,
        "event_flag": event_flag,
        "area_m2": room_info[room_id]["area_m2"],
        "lag_1h_energy": 0,
        "roll_6h_mean_energy": 0,
        "room_type_lab": room_info[room_id]["room_type_lab"],
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "sin_dayofweek": sin_dayofweek,
        "cos_dayofweek": cos_dayofweek,
    }

# ------------------- Prediction -------------------
if mode == "specific_hour":
    df_input = pd.DataFrame([generate_features(hour, occupancy)])
    X_scaled = scaler_X.transform(df_input)

    # Create dummy 168-length sequence
    X_seq = np.repeat(X_scaled[np.newaxis, :, :], 168, axis=1)
    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    predicted_kWh = float(y_pred.flatten()[0])
    carbon_footprint = predicted_kWh * 0.82

    print(f"\n🔹 Predicted electricity usage for room {room_id} at {hour:02d}:00 → {predicted_kWh:.2f} kWh")
    print(f"🌱 Estimated carbon footprint: {carbon_footprint:.2f} kg CO₂")

else:
    # Whole day prediction hour by hour
    predicted_values = []
    for h in range(24):
        df_hour = pd.DataFrame([generate_features(h, occupancy)])
        X_scaled = scaler_X.transform(df_hour)
        X_seq = np.repeat(X_scaled[np.newaxis, :, :], 168, axis=1)
        y_pred_scaled = model.predict(X_seq, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        predicted_values.append(float(y_pred.flatten()[0]))

    y_pred_flat = np.array(predicted_values)

    print(f"\n🔹 Predicted electricity usage for room {room_id} on {date} (Whole Day):")
    print("---------------------------------------------------------------")
    print(f"{'Hour':<10}{'Predicted Usage (kWh)':>25}")
    print("---------------------------------------------------------------")

    for h, val in enumerate(y_pred_flat):
        print(f"{h:02d}:00{' ' * 10}{val:>15.2f}")

    print("---------------------------------------------------------------")

    # ---- Calculate metrics ----
    peak_hour = np.argmax(y_pred_flat)
    peak_value = y_pred_flat[peak_hour]
    total_energy = np.sum(y_pred_flat)
    carbon_footprint_total = total_energy * 0.82

    print(f"\n⚡ Peak hour: {peak_hour:02d}:00 with {peak_value:.2f} kWh")
    print(f"🔋 Total energy usage: {total_energy:.2f} kWh")
    print(f"🌱 Total estimated carbon footprint: {carbon_footprint_total:.2f} kg CO₂")

    # ---- Visualization ----
    plt.figure(figsize=(8, 4))
    plt.bar(range(24), y_pred_flat, color="skyblue", edgecolor="black")
    plt.title(f"Electricity Usage Prediction for {room_id} on {date}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Predicted Energy (kWh)")
    plt.xticks(range(24), [f"{h:02d}:00" for h in range(24)], rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

print("\n✅ Prediction completed successfully.")
