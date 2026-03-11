# ============================================================
# Energy Usage Prediction - Streamlit App (Centered Compact UI)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import os
from datetime import datetime

# ------------------- Page Config -------------------
st.set_page_config(page_title="Energy Usage Prediction", layout="wide")

# ------------------- Custom CSS -------------------
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
        font-family: 'Segoe UI', sans-serif;
    }
    .container {
        max-width: 700px;
        margin: auto;
        background-color: #ffffff;
        padding: 30px 40px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: 700;
        color: #0b5394;
        margin-bottom: 10px;
    }
    .overview-box {
        background-color: #eaf0f6;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #b3c6e0;
        margin-bottom: 25px;
        text-align: justify;
    }
    .section-title {
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        color: #0b5394;
        margin-top: 25px;
    }
    .stButton>button {
        background-color: #0b5394;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 25px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #073b6a;
        color: white;
    }
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stDateInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 1px solid #b3c6e0;
        padding: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Title -------------------
st.markdown("<div class='title'>Energy Usage Prediction</div>", unsafe_allow_html=True)

# ------------------- Centered Container -------------------
with st.container():
    st.markdown("<div class='container'>", unsafe_allow_html=True)

    # Overview Section
    st.markdown("""
    <div class="overview-box">
        This project predicts electricity consumption in campus rooms using a <b>CNN + BiLSTM deep learning model</b>. 
        It analyzes occupancy, outdoor temperature, solar irradiance, and day patterns 
        to estimate hourly or daily energy usage. It also calculates the associated 
        <b>carbon footprint</b>, helping facilities optimize power utilization and support sustainability goals.
    </div>
    """, unsafe_allow_html=True)

    # Load Model and Scalers
    @st.cache_resource
    def load_artifacts():
        base_path = os.path.join(os.getcwd(), "saved_cnn_bilstm_model")
        model = load_model(os.path.join(base_path, "cnn_bilstm_electricity_model.h5"), compile=False)
        scaler_X = joblib.load(os.path.join(base_path, "scaler_X.pkl"))
        scaler_y = joblib.load(os.path.join(base_path, "scaler_y.pkl"))
        return model, scaler_X, scaler_y

    model, scaler_X, scaler_y = load_artifacts()

    # Room Info
    room_info = {
        "C101": {"area_m2": 50, "room_type_lab": 0},
        "C102": {"area_m2": 60, "room_type_lab": 0},
        "L201": {"area_m2": 70, "room_type_lab": 1},
        "L202": {"area_m2": 80, "room_type_lab": 1},
    }

    # Feature Generator
    def generate_features(hour, date, occupancy, exam_flag, event_flag, room_id):
        dayofweek = pd.to_datetime(date).weekday()
        is_weekend = int(dayofweek >= 5)
        day_of_year = pd.to_datetime(date).dayofyear
        outdoor_temp_C = 20 + 10 * np.sin((day_of_year / 365) * 2 * np.pi)
        solar_irradiance_Wm2 = max(0, 800 * np.sin((hour / 24) * np.pi) * np.sin((day_of_year / 365) * np.pi))
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

    # Prediction Mode Selection
    st.markdown("<div class='section-title'>Select Prediction Mode</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict for Whole Day"):
            st.session_state["selected_mode"] = "whole_day"
    with col2:
        if st.button("Predict for Specific Hour"):
            st.session_state["selected_mode"] = "specific_hour"

    # Input Form
    if "selected_mode" in st.session_state:
        st.markdown("<div class='section-title'>Enter Input Details</div>", unsafe_allow_html=True)

        date = st.date_input("Select Date", datetime.now())
        room_id = st.selectbox("Select Room ID", list(room_info.keys()))
        event_flag = st.selectbox("Event Flag (0 = No Event, 1 = Event)", [0, 1])
        exam_flag = st.selectbox("Exam Flag (0 = No Exam, 1 = Exam)", [0, 1])
        occupancy = st.number_input("Occupancy (Number of People)", min_value=0, max_value=100, value=10)

        if st.session_state["selected_mode"] == "specific_hour":
            hour = st.slider("Select Hour (0 - 23)", 0, 23, 12)

        if st.button("Generate Prediction"):
            st.session_state["date"] = date
            st.session_state["room_id"] = room_id
            st.session_state["event_flag"] = event_flag
            st.session_state["exam_flag"] = exam_flag
            st.session_state["occupancy"] = occupancy
            st.session_state["hour"] = hour if st.session_state["selected_mode"] == "specific_hour" else None
            st.session_state["predict"] = True
            st.rerun()

    # Results Section
    if "predict" in st.session_state and st.session_state["predict"]:
        st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)

        date = st.session_state["date"]
        room_id = st.session_state["room_id"]
        event_flag = st.session_state["event_flag"]
        exam_flag = st.session_state["exam_flag"]
        occupancy = st.session_state["occupancy"]

        if st.session_state["selected_mode"] == "specific_hour":
            hour = st.session_state["hour"]
            df_input = pd.DataFrame([generate_features(hour, date, occupancy, exam_flag, event_flag, room_id)])
            X_scaled = scaler_X.transform(df_input)
            X_seq = np.repeat(X_scaled[np.newaxis, :, :], 168, axis=1)
            y_pred_scaled = model.predict(X_seq, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            predicted_kWh = float(y_pred.flatten()[0])
            carbon_footprint = predicted_kWh * 0.82

            st.metric("Predicted Energy Usage (kWh)", f"{predicted_kWh:.2f}")
            st.metric("Estimated Carbon Footprint (kg CO₂)", f"{carbon_footprint:.2f}")

        else:
            predicted_values = []
            for h in range(24):
                df_hour = pd.DataFrame([generate_features(h, date, occupancy, exam_flag, event_flag, room_id)])
                X_scaled = scaler_X.transform(df_hour)
                X_seq = np.repeat(X_scaled[np.newaxis, :, :], 168, axis=1)
                y_pred_scaled = model.predict(X_seq, verbose=0)
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
                predicted_values.append(float(y_pred.flatten()[0]))

            y_pred_flat = np.array(predicted_values)
            peak_hour = np.argmax(y_pred_flat)
            total_energy = np.sum(y_pred_flat)
            carbon_footprint_total = total_energy * 0.82

            colA, colB, colC = st.columns(3)
            colA.metric("Peak Hour", f"{peak_hour:02d}:00")
            colB.metric("Total Energy (kWh)", f"{total_energy:.2f}")
            colC.metric("Total Carbon Footprint (kg CO₂)", f"{carbon_footprint_total:.2f}")

            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(range(24), y_pred_flat, marker='o', color='#0b5394', linewidth=2)
            ax.fill_between(range(24), y_pred_flat, color='#b3c6e0', alpha=0.4)
            ax.set_title(f"Hourly Energy Prediction - {room_id} ({date})", fontsize=12, fontweight='bold')
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Predicted Energy (kWh)")
            ax.set_xticks(range(24))
            ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            if st.button("Back to Home"):
                st.session_state["predict"] = False
                st.session_state["selected_mode"] = None
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
