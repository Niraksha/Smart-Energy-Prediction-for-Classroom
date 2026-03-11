# ============================================================
# LSTM Electricity Usage Prediction - Local VS Code Version (With Carbon Footprint)
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os

# ------------------- User Inputs -------------------
mode = input("Prediction mode (whole_day / specific_hour): ").strip().lower()
date = input("Enter date (YYYY-MM-DD): ").strip()
room_id = input("Enter room ID (e.g., C101): ").strip()
event_flag = int(input("Enter event flag (0/1): ").strip())
exam_flag = int(input("Enter exam flag (0/1): ").strip())
occupancy = int(input("Enter occupancy (number of people): ").strip())

if mode == "specific_hour":
    hour = int(input("Enter hour (0-23): ").strip())

# ------------------- Load saved model and scalers -------------------
base_path = os.path.join(os.getcwd(), "saved_lstm_model")

model_path = os.path.join(base_path, "lstm_electricity_model.h5")
scaler_X_path = os.path.join(base_path, "scaler_X.pkl")
scaler_y_path = os.path.join(base_path, "scaler_y.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")
if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
    raise FileNotFoundError("❌ Scaler files not found. Make sure training was completed.")

model = load_model(model_path, compile=False)
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

print("✅ Model and scalers loaded successfully!\n")

# ------------------- Room info -------------------
room_info = {
    "C101": {"area_m2": 50, "room_type_lab": 0},
    "C102": {"area_m2": 60, "room_type_lab": 0},
    "L201": {"area_m2": 70, "room_type_lab": 1},
    "L202": {"area_m2": 80, "room_type_lab": 1},
}

if room_id not in room_info:
    raise ValueError(f"Invalid room ID! Choose from {list(room_info.keys())}")

# ------------------- Derived feature generator -------------------
def generate_features(hour, occupancy):
    dayofweek = pd.to_datetime(date).weekday()
    is_weekend = int(dayofweek >= 5)
    day_of_year = pd.to_datetime(date).dayofyear

    # Approximate environmental features
    outdoor_temp_C = 20 + 10 * np.sin((day_of_year / 365) * 2 * np.pi)
    solar_irradiance_Wm2 = max(0, 800 * np.sin((hour / 24) * np.pi) * np.sin((day_of_year / 365) * np.pi))

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
    }

# ------------------- Build input dataframe -------------------
if mode == "specific_hour":
    data_list = [generate_features(hour, occupancy)]
elif mode == "whole_day":
    data_list = [generate_features(h, occupancy) for h in range(24)]
else:
    raise ValueError("Invalid mode! Must be 'whole_day' or 'specific_hour'.")

df_input = pd.DataFrame(data_list)

# ------------------- Scale and prepare for LSTM -------------------
X_scaled = scaler_X.transform(df_input)
time_steps = 48

if len(X_scaled) < time_steps:
    repeat = time_steps - len(X_scaled)
    pad = np.repeat(X_scaled[:1, :][np.newaxis, :, :], repeat, axis=1)
    X_seq = np.concatenate([pad, X_scaled[np.newaxis, :, :]], axis=1)
else:
    X_seq = np.array([X_scaled[-time_steps:]])

# ------------------- Predict -------------------
y_pred_scaled = model.predict(X_seq)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# ------------------- Results -------------------
if mode == "specific_hour":
    predicted_kWh = float(y_pred.flatten()[0])  # ensure scalar
    carbon_footprint = predicted_kWh * 0.82     # kg CO₂ per kWh

    print(f"\n🔹 Predicted electricity usage for room {room_id} at hour {hour}: {predicted_kWh:.2f} kWh")
    print(f"🌱 Estimated carbon footprint: {carbon_footprint:.2f} kg CO₂")

else:
    print(f"\n🔹 Predicted electricity usage for room {room_id} (whole day):")
    for h, val in enumerate(y_pred):
        print(f"  Hour {h:02d}: {val[0]:.2f} kWh")

    peak_hour = np.argmax(y_pred)
    total_energy = np.sum(y_pred)
    carbon_footprint_total = total_energy * 0.82  # kg CO₂ per kWh

    print(f"\n⚡ Peak hour: {peak_hour:02d} with {y_pred[peak_hour][0]:.2f} kWh")
    print(f"🌱 Total estimated carbon footprint for {date}: {carbon_footprint_total:.2f} kg CO₂")

print("\n✅ Prediction completed successfully.")
