# ============================================================
# Improved LSTM Electricity Usage Prediction - Bidirectional + Time Encoding
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import math
import os
import sys
import matplotlib.pyplot as plt

# ============================================================
# 1) Load dataset
# ============================================================

data_path = "synthetic_campus_energy_2022_2025.csv"

if not os.path.exists(data_path):
    print(f"❌ Dataset not found at: {data_path}")
    sys.exit(1)

df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully!")
print(df.head())

# ============================================================
# 2) Feature Engineering - Time-based Cyclic Encoding
# ============================================================

# Encode hour and dayofweek as cyclic features
df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

print("✅ Added cyclic time encoding (sin/cos of hour & dayofweek).")

# ============================================================
# 3) One-hot encode room_type column (if exists)
# ============================================================

if "room_type" in df.columns:
    df = pd.get_dummies(df, columns=["room_type"], drop_first=True)
    print("✅ One-hot encoding done for room_type.")

# ============================================================
# 4) Feature selection
# ============================================================

features = [
    "hour", "dayofweek", "is_weekend", "occupancy", "outdoor_temp_C",
    "solar_irradiance_Wm2", "exam_flag", "event_flag", "area_m2",
    "lag_1h_energy", "roll_6h_mean_energy",
    "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek"
]

features += [col for col in df.columns if col.startswith("room_type_")]
target = "energy_kWh"

if target not in df.columns:
    print("❌ Target column 'energy_kWh' not found.")
    sys.exit(1)

X = df[features].values
y = df[target].values

# ============================================================
# 5) Scale features
# ============================================================

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# ============================================================
# 6) Create sequences for LSTM
# ============================================================

def create_sequences(X, y, time_steps=168):  # 1 week context
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i - time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

time_steps = 168
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
print(f"✅ Sequence data shape: {X_seq.shape}, {y_seq.shape}")

# ============================================================
# 7) Train-test split
# ============================================================

split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# ============================================================
# 8) Build Bidirectional LSTM Model
# ============================================================

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(32),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ============================================================
# 9) Callbacks
# ============================================================

early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=7, factor=0.5, min_lr=1e-5)

# ============================================================
# 10) Train model
# ============================================================

print("\n🚀 Training improved Bidirectional LSTM model... please wait.\n")

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n✅ Training complete!")

# ============================================================
# 11) Evaluate Model
# ============================================================

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = math.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("\n📊 Improved LSTM Model Performance Metrics:")
print("---------------------------------------------------")
print(f"✅ Mean Absolute Error (MAE):      {mae:.3f} kWh")
print(f"✅ Root Mean Square Error (RMSE):  {rmse:.3f} kWh")
print(f"✅ Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"✅ R² Score:                       {r2:.4f}")
print("---------------------------------------------------")

# ============================================================
# 12) Visualization
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(y_true[:200], label="Actual", linewidth=2)
plt.plot(y_pred[:200], label="Predicted", linewidth=2, linestyle="--")
plt.title("Improved Bidirectional LSTM - Actual vs Predicted (Sample 200 points)")
plt.xlabel("Time Steps")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 13) Save Model and Scalers
# ============================================================

save_path = "saved_lstm_model_improved"
os.makedirs(save_path, exist_ok=True)

model.save(os.path.join(save_path, "lstm_electricity_model_improved.h5"))
joblib.dump(scaler_X, os.path.join(save_path, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(save_path, "scaler_y.pkl"))

print("\n💾 Improved model and scalers saved successfully!")
print(f"📂 Files saved in: {os.path.abspath(save_path)}")

# ============================================================
# 14) Summary
# ============================================================

print("\n✅ Improved Bidirectional LSTM completed successfully!")
print(f"📊 Final Metrics → R²: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
