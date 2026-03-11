# ============================================================
# Evaluate Saved Improved Bidirectional LSTM Model
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import math
import os
import matplotlib.pyplot as plt

# ============================================================
# 1) Paths - Update if needed
# ============================================================

base_path = os.path.join(os.getcwd(), "saved_lstm_model_improved")
model_path = os.path.join(base_path, "lstm_electricity_model_improved.h5")
scaler_X_path = os.path.join(base_path, "scaler_X.pkl")
scaler_y_path = os.path.join(base_path, "scaler_y.pkl")
data_path = "synthetic_campus_energy_2022_2025.csv"  # 👈 Your dataset

# ============================================================
# 2) Load Model and Scalers
# ============================================================

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at: {model_path}")
if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
    raise FileNotFoundError("❌ Scalers not found. Please train the improved model first.")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at: {data_path}")

print("✅ Loading saved model and scalers...")
model = load_model(model_path, compile=False)
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)
print("✅ Model and scalers loaded successfully!\n")

# ============================================================
# 3) Load Dataset for Evaluation
# ============================================================

df = pd.read_csv(data_path)
print("✅ Dataset loaded for evaluation!")

# ============================================================
# 4) Recreate Feature Engineering (same as training)
# ============================================================

# Add cyclic time encodings
df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

if "room_type" in df.columns:
    df = pd.get_dummies(df, columns=["room_type"], drop_first=True)

# Define same features as training
features = [
    "hour", "dayofweek", "is_weekend", "occupancy", "outdoor_temp_C",
    "solar_irradiance_Wm2", "exam_flag", "event_flag", "area_m2",
    "lag_1h_energy", "roll_6h_mean_energy",
    "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek"
]
features += [col for col in df.columns if col.startswith("room_type_")]
target = "energy_kWh"

if target not in df.columns:
    raise ValueError("❌ Target column 'energy_kWh' not found in dataset.")

X = df[features].values
y = df[target].values

X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y.reshape(-1, 1))

# ============================================================
# 5) Create Sequences (same as in training)
# ============================================================

def create_sequences(X, y, time_steps=168):
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i - time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

time_steps = 168
X_seq, y_seq = create_sequences(X_scaled, y_scaled)
print(f"✅ Sequence data prepared: {X_seq.shape}\n")

# ============================================================
# 6) Evaluate Model
# ============================================================

print("🔍 Evaluating saved improved LSTM model (no training)...")
y_pred_scaled = model.predict(X_seq, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_seq)

# ============================================================
# 7) Compute Metrics
# ============================================================

mae = mean_absolute_error(y_true, y_pred)
rmse = math.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("\n📊 Saved Improved LSTM Model Performance Metrics:")
print("---------------------------------------------------")
print(f"✅ Mean Absolute Error (MAE):      {mae:.3f} kWh")
print(f"✅ Root Mean Square Error (RMSE):  {rmse:.3f} kWh")
print(f"✅ Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"✅ R² Score:                       {r2:.4f}")
print("---------------------------------------------------")

# ============================================================
# 8) Visualization
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(y_true[:200], label="Actual", linewidth=2)
plt.plot(y_pred[:200], label="Predicted", linewidth=2, linestyle="--")
plt.title("Improved LSTM - Actual vs Predicted (Sample 200 points)")
plt.xlabel("Time Steps")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 9) Save Metrics Summary
# ============================================================

metrics_df = pd.DataFrame({
    "MAE (kWh)": [mae],
    "RMSE (kWh)": [rmse],
    "MAPE (%)": [mape],
    "R² Score": [r2]
})

metrics_file = os.path.join(base_path, "saved_model_performance.csv")
metrics_df.to_csv(metrics_file, index=False)

print(f"\n💾 Metrics saved successfully → {metrics_file}")

# ============================================================
# 10) Summary
# ============================================================

print("\n✅ Evaluation Complete — Saved Model Performance Summary:")
print(f"📊 R² Score: {r2:.4f}")
print(f"📉 MAE: {mae:.2f} kWh")
print(f"📉 RMSE: {rmse:.2f} kWh")
print(f"📈 MAPE: {mape:.2f}%")
