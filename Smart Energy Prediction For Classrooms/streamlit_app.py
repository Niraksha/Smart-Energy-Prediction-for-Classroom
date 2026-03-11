import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime, date
import time

# Page configuration
st.set_page_config(
    page_title="Energy Usage Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS styling with hover effects
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
        padding-top: 1rem;
    }
    
    .model-overview {
        background: white;
        border: 2px solid #3498db;
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .model-overview:hover {
        border-color: #2980b9;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2);
        transform: translateY(-2px);
    }
    
    .compact-input {
        background: white;
        border: 2px solid #ecf0f1;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .compact-input:hover {
        border-color: #3498db;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.15);
    }
    
    .result-box {
        background: white;
        border: 3px solid #3498db;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .result-box:hover {
        border-color: #2980b9;
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.25);
        transform: translateY(-3px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1.1rem;
        color: #34495e;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .stButton > button {
        background: #3498db;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        width: 100%;
        margin-top: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    
    .stats-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #2c3e50;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .stats-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .chart-container {
        background: white;
        border: 2px solid #ecf0f1;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        border-color: #3498db;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.15);
    }
    
    .analysis-box {
        background: white;
        border: 2px solid #95a5a6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .analysis-box:hover {
        border-color: #3498db;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

@st.cache_resource
def load_model_and_scalers():
    """Load the trained model and scalers"""
    try:
        base_path = os.path.join(os.getcwd(), "saved_lstm_model_improved")
        
        model_path = os.path.join(base_path, "lstm_electricity_model_improved.h5")
        scaler_X_path = os.path.join(base_path, "scaler_X.pkl")
        scaler_y_path = os.path.join(base_path, "scaler_y.pkl")
        
        model = load_model(model_path, compile=False)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        
        return model, scaler_X, scaler_y, True
    except Exception as e:
        return None, None, None, False

def generate_features(hour, occupancy, date_input, room_id, event_flag, exam_flag):
    """Generate features for prediction"""
    room_info = {
        "C101": {"area_m2": 50, "room_type_lab": 0},
        "C102": {"area_m2": 60, "room_type_lab": 0},
        "L201": {"area_m2": 70, "room_type_lab": 1},
        "L202": {"area_m2": 80, "room_type_lab": 1},
    }
    
    dayofweek = pd.to_datetime(date_input).weekday()
    is_weekend = int(dayofweek >= 5)
    day_of_year = pd.to_datetime(date_input).dayofyear

    # Environmental features
    outdoor_temp_C = 20 + 10 * np.sin((day_of_year / 365) * 2 * np.pi)
    solar_irradiance_Wm2 = max(0, 800 * np.sin((hour / 24) * np.pi) * np.sin((day_of_year / 365) * np.pi))

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

def make_prediction(model, scaler_X, scaler_y, features_df):
    """Make prediction using the loaded model"""
    X_scaled = scaler_X.transform(features_df)
    time_steps = 168
    
    if len(X_scaled) < time_steps:
        repeat = time_steps - len(X_scaled)
        pad = np.repeat(X_scaled[:1, :][np.newaxis, :, :], repeat, axis=1)
        X_seq = np.concatenate([pad, X_scaled[np.newaxis, :, :]], axis=1)
    else:
        X_seq = np.array([X_scaled[-time_steps:]])
    
    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    return y_pred

def create_hourly_chart(predictions, room_id):
    """Create a clean hourly prediction chart"""
    hours = list(range(24))
    values = predictions.flatten()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=values,
        mode='lines+markers',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6, color='#2980b9'),
        hovertemplate='<b>%{x}:00</b><br>%{y:.2f} kWh<extra></extra>',
        name='Energy Usage'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'24-Hour Energy Prediction - Room {room_id}',
            font=dict(size=16, color='#2c3e50'),
            x=0.5
        ),
        xaxis_title='Hour of Day',
        yaxis_title='Energy Consumption (kWh)',
        template='plotly_white',
        height=400,
        showlegend=False,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig

def calculate_usage_analysis(predictions):
    """Calculate detailed usage analysis"""
    values = predictions.flatten()
    avg_usage = np.mean(values)
    
    # Low usage hours (below 80% of average)
    low_threshold = avg_usage * 0.8
    low_usage_hours = [(h, v) for h, v in enumerate(values) if v <= low_threshold]
    
    # Peak usage hours (above 120% of average)
    peak_threshold = avg_usage * 1.2
    peak_usage_hours = [(h, v) for h, v in enumerate(values) if v >= peak_threshold]
    
    # Calculate total low and peak energy
    total_low_energy = sum([v for _, v in low_usage_hours])
    total_peak_energy = sum([v for _, v in peak_usage_hours])
    
    return {
        'low_hours': low_usage_hours,
        'peak_hours': peak_usage_hours,
        'total_low_energy': total_low_energy,
        'total_peak_energy': total_peak_energy,
        'avg_usage': avg_usage,
        'low_threshold': low_threshold,
        'peak_threshold': peak_threshold
    }

# Main App
def main():
    # Title
    st.markdown('<h1 style="text-align: center; font-size: 2.5rem; font-weight: 600; color: #2c3e50; margin-bottom: 2rem;">Energy Usage Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model
    model, scaler_X, scaler_y, model_loaded = load_model_and_scalers()
    
    if not model_loaded:
        st.error("Failed to load the trained model. Please ensure model files exist.")
        return
    
    # Model Overview Section
    with st.container():
        st.markdown("""
        <div class="model-overview">
            <h2 style="color: #2c3e50; margin-bottom: 1.5rem;">Model Architecture Overview</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Input Features (15 parameters):**
            - **Temporal**: Hour, Day of Week, Weekend Flag
            - **Environmental**: Outdoor Temperature, Solar Irradiance  
            - **Occupancy**: Number of People
            - **Events**: Event Flag, Exam Flag
            - **Room**: Area (m²), Room Type (Lab/Classroom)
            - **Advanced**: Lag Features, Rolling Averages, Cyclic Encodings
            """)
            
        with col2:
            st.markdown("""
            **LSTM Architecture:**
            - **Model Type**: Bidirectional LSTM
            - **Sequence Length**: 168 time steps (1 week)
            - **Hidden Layers**: 64 → 32 neurons
            - **Activation**: ReLU, Dropout (0.2)
            - **Output**: Energy consumption (kWh)
            """)
    
    # Performance metrics - 4 horizontal boxes
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-container">
            <div style="font-size: 1.5rem; font-weight: 600;">91.6%</div>
            <div style="font-size: 0.9rem;">R² Score</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="stats-container">
            <div style="font-size: 1.5rem; font-weight: 600;">2.11</div>
            <div style="font-size: 0.9rem;">MAE (kWh)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="stats-container">
            <div style="font-size: 1.5rem; font-weight: 600;">2.77</div>
            <div style="font-size: 0.9rem;">RMSE (kWh)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="stats-container">
            <div style="font-size: 1.5rem; font-weight: 600;">7.33%</div>
            <div style="font-size: 0.9rem;">MAPE</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Compact Input Section
    st.markdown("""
    <div class="compact-input">
        <h3 style="color: #2c3e50; text-align: center; margin-bottom: 1rem;">Prediction Parameters</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        col_a, col_b = st.columns(2)
        with col_a:
            prediction_mode = st.selectbox("Mode", ["Specific Hour", "Whole Day"])
            selected_date = st.date_input("Date", value=date.today())
            room_id = st.selectbox("Room", ["C101", "C102", "L201", "L202"])
            
        with col_b:
            occupancy = st.number_input("Occupancy", min_value=0, max_value=100, value=30)
            event_flag = st.selectbox("Event", [0, 1], format_func=lambda x: "Yes" if x else "No")
            exam_flag = st.selectbox("Exam", [0, 1], format_func=lambda x: "Yes" if x else "No")
            if prediction_mode == "Specific Hour":
                hour = st.selectbox("Hour", list(range(24)), index=12)
        
        predict_button = st.button("Generate Prediction")
    
    # Show results only after prediction button is clicked
    if predict_button:
        st.session_state.prediction_made = True
    
    if st.session_state.prediction_made:
        # Results section
        st.markdown("---")
        
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.005)
        progress_bar.empty()
        
        try:
            if prediction_mode == "Specific Hour":
                features = generate_features(hour, occupancy, selected_date, room_id, event_flag, exam_flag)
                df_input = pd.DataFrame([features])
                
                prediction = make_prediction(model, scaler_X, scaler_y, df_input)
                predicted_kWh = float(prediction.flatten()[0])
                carbon_footprint = predicted_kWh * 0.82
                
                st.markdown("<h2 style='text-align: center; color: #2c3e50; margin: 2rem 0;'>Prediction Results</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="metric-label">Energy Consumption</div>
                        <div class="metric-value">{predicted_kWh:.2f} kWh</div>
                        <div style="color: #7f8c8d; margin-top: 1rem;">
                            Room {room_id} • {hour}:00 • {selected_date.strftime('%B %d, %Y')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="metric-label">Carbon Footprint</div>
                        <div class="metric-value">{carbon_footprint:.2f} kg CO₂</div>
                        <div style="color: #7f8c8d; margin-top: 1rem;">
                            Environmental Impact
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                features_list = [generate_features(h, occupancy, selected_date, room_id, event_flag, exam_flag) for h in range(24)]
                df_input = pd.DataFrame(features_list)
                
                predictions = make_prediction(model, scaler_X, scaler_y, df_input)
                
                peak_hour = np.argmax(predictions)
                total_energy = np.sum(predictions)
                carbon_footprint_total = total_energy * 0.82
                
                st.markdown("<h2 style='text-align: center; color: #2c3e50; margin: 2rem 0;'>24-Hour Energy Analysis</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="metric-label">Total Energy Consumption</div>
                        <div class="metric-value">{total_energy:.1f} kWh</div>
                        <div style="color: #7f8c8d; margin-top: 1rem;">
                            Peak: {peak_hour}:00 ({predictions[peak_hour][0]:.1f} kWh)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="metric-label">Total Carbon Footprint</div>
                        <div class="metric-value">{carbon_footprint_total:.1f} kg CO₂</div>
                        <div style="color: #7f8c8d; margin-top: 1rem;">
                            24-hour Environmental Impact
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Chart
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                chart = create_hourly_chart(predictions, room_id)
                st.plotly_chart(chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed Usage Analysis
                analysis = calculate_usage_analysis(predictions)
                
                st.markdown("<h3 style='color: #2c3e50; margin: 2rem 0 1rem 0;'>Usage Pattern Analysis</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="analysis-box">
                        <h4 style="color: #27ae60; margin-bottom: 1rem;">Low Energy Usage Hours</h4>
                        <p><strong>Threshold:</strong> ≤ {analysis['low_threshold']:.1f} kWh</p>
                        <p><strong>Total Hours:</strong> {len(analysis['low_hours'])}</p>
                        <p><strong>Total Energy:</strong> {analysis['total_low_energy']:.1f} kWh</p>
                        <p><strong>Hours:</strong> {', '.join([f"{h}:00 ({v:.1f})" for h, v in analysis['low_hours'][:6]])}</p>
                        {f"<p><em>...and {len(analysis['low_hours'])-6} more</em></p>" if len(analysis['low_hours']) > 6 else ""}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="analysis-box">
                        <h4 style="color: #e74c3c; margin-bottom: 1rem;">Peak Energy Usage Hours</h4>
                        <p><strong>Threshold:</strong> ≥ {analysis['peak_threshold']:.1f} kWh</p>
                        <p><strong>Total Hours:</strong> {len(analysis['peak_hours'])}</p>
                        <p><strong>Total Energy:</strong> {analysis['total_peak_energy']:.1f} kWh</p>
                        <p><strong>Hours:</strong> {', '.join([f"{h}:00 ({v:.1f})" for h, v in analysis['peak_hours'][:6]])}</p>
                        {f"<p><em>...and {len(analysis['peak_hours'])-6} more</em></p>" if len(analysis['peak_hours']) > 6 else ""}
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 2rem;">
        <p>Powered by Bidirectional LSTM Neural Network</p>
        <p>Built with Streamlit • TensorFlow • Plotly</p>
    </div>
    """, unsafe_allow_html=True)