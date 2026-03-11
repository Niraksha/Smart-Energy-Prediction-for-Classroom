# Smart-Energy-Prediction-for-Classroom
AI-based energy consumption prediction system for classrooms using MLP and CNN-BiLSTM models to forecast electricity usage based on environmental, occupancy, and time-based factors.

This project focuses on predicting electricity consumption in classrooms using Artificial Intelligence and Deep Learning techniques. The goal is to build a predictive system that forecasts energy usage based on environmental conditions, occupancy patterns, and time-based features.

Accurate energy forecasting helps institutions optimize electricity usage, reduce wastage, and support sustainable energy management.

## Project Overview

Energy consumption in buildings varies depending on multiple factors such as:

- Occupancy levels
- Temperature conditions
- Solar irradiance
- Time of day
- Room type and size

This project compares two machine learning models:

1. **Multi-Layer Perceptron (MLP)** – a traditional neural network used as a baseline model.
2. **CNN + BiLSTM Hybrid Model** – an advanced deep learning architecture designed to capture both short-term patterns and long-term temporal dependencies in energy usage data.

## Dataset

The dataset used is a synthetic campus energy dataset covering hourly energy usage data between **2022 and 2025**.

### Features included

#### Time-Based Features
- hour
- dayofweek
- is_weekend
- lag_1h_energy
- roll_6h_mean_energy

These help capture daily and weekly energy consumption patterns.

#### Environmental Features
- outdoor_temp_C
- solar_irradiance_Wm2

These influence electricity usage such as cooling systems and lighting.

#### Occupancy and Room Features
- occupancy
- area_m2
- room_type_lab

These describe how the number of people and room size affect energy consumption.

#### Event Features
- exam_flag
- event_flag

Special events or exams can create unusual spikes in electricity usage.

#### Target Variable
- energy_kWh (energy consumption)

## Data Preprocessing

Before training the models, the following preprocessing steps were performed:

- Min-Max normalization
- One-hot encoding for categorical variables
- Sinusoidal encoding for cyclical features such as hour and weekday
- Feature scaling and sequence generation for time-series modeling

## Model Architectures

### 1. Multi-Layer Perceptron (MLP)

The MLP model acts as the baseline model and learns non-linear relationships between features and energy usage.

Architecture:

- Input layer (13 features)
- Dense layer (128 neurons, ReLU)
- Batch Normalization
- Dropout (0.3)
- Dense layer (64 neurons, ReLU)
- Batch Normalization + Dropout (0.2)
- Dense layer (32 neurons)
- Output layer (1 neuron – energy prediction)

### 2. CNN + BiLSTM Model

The hybrid model combines CNN and Bidirectional LSTM layers.

Purpose:

- CNN extracts short-term energy patterns.
- BiLSTM captures long-term temporal dependencies.

Architecture:

- Conv1D layers for local temporal feature extraction
- MaxPooling layers
- Batch Normalization
- Dropout for regularization
- Bidirectional LSTM layers
- Dense layers for final prediction
- Output layer predicting next-hour energy usage

## Model Performance

### MLP Model

| Metric | Value |
|------|------|
| MAE | 1.62 kWh |
| RMSE | 2.03 kWh |
| R² Score | 0.969 |

### CNN + BiLSTM Model

| Metric | Value |
|------|------|
| MAE | 2.10 kWh |
| RMSE | 2.76 kWh |
| R² Score | 0.9167 |

The MLP model achieved slightly better numerical accuracy, while the CNN + BiLSTM model captured temporal energy usage patterns more effectively.


## Model Comparison

| Model | MAE | RMSE | R² Score | Best For |
|------|------|------|------|------|
| MLP | 1.62 | 2.03 | 0.969 | Static feature analysis |
| CNN + BiLSTM | 2.10 | 2.76 | 0.9167 | Time-series forecasting |


## Applications

This system can be used in:

- Smart campus energy management
- Building energy optimization
- Electricity demand forecasting
- Sustainable infrastructure planning
- Smart city energy analytics

## Key Insights

- Traditional neural networks perform well for static data.
- Temporal deep learning models capture dynamic energy trends.
- Combining environmental, occupancy, and time-based features improves prediction accuracy.

## Future Improvements

Potential improvements include:

- Integration with real-time IoT sensor data
- Smart building automation systems
- Deployment as a web-based dashboard
- Carbon footprint estimation based on energy consumption
