# ------------------- Improved Traditional ANN (MLP) -------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ------------------- 1) Load dataset -------------------
df = pd.read_csv("synthetic_campus_energy_2022_2025.csv")

# ------------------- 2) Preprocess features -------------------
df = pd.get_dummies(df, columns=["room_type"], drop_first=True)

# Feature engineering
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

features = ['hour_sin','hour_cos','dayofweek','is_weekend','occupancy','outdoor_temp_C',
            'solar_irradiance_Wm2','exam_flag','event_flag','area_m2',
            'lag_1h_energy','roll_6h_mean_energy','room_type_lab']
target = 'energy_kWh'

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

X = df[features].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------- 3) Build improved MLP model -------------------
mlp = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(1)
])

mlp.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ------------------- 4) Callbacks -------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-5)

# ------------------- 5) Train model -------------------
history = mlp.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ------------------- 6) Evaluate model -------------------
loss, mae = mlp.evaluate(X_test, y_test)
y_pred = mlp.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Improved MLP Performance:")
print(f"MAE: {mae:.2f} kWh | RMSE: {rmse:.2f} kWh | R²: {r2:.3f}")

# ------------------- 7) Visualizations -------------------
plt.figure(figsize=(10,5))
plt.plot(y_test[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.title("Improved MLP - Actual vs Predicted Energy Usage")
plt.xlabel("Samples")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.grid(True)
plt.show()