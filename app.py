import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Sensor Anomaly Detection",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "model/anomaly_model.pkl"
DATA_PATH = "data/sensor_predictions.csv"
METRICS_PATH = "model/metrics.json"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Trained model not found. Please run train_model.py first.")
    st.stop()

# ============================================================
# LOAD MODEL METRICS
# ============================================================

try:
    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

except FileNotFoundError:
    st.error(
        "Model metrics not found. "
        "Please run train_model.py first."
    )
    st.stop()

# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(
        "Prediction dataset not found. "
        "Please run train_model.py first."
    )
    st.stop()


df["timestamp"] = pd.to_datetime(df["timestamp"])


# ============================================================
# SESSION STATE FOR LIVE READINGS
# ============================================================

if "live_readings" not in st.session_state:

    st.session_state.live_readings = pd.DataFrame(
        columns=[
            "timestamp",
            "temperature",
            "vibration",
            "pressure",
            "humidity",
            "prediction"
        ]
    )


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Sensor Anomaly Detection System")

st.markdown(
    """
    **Industrial Sensor Monitoring Dashboard**

    Machine-learning powered anomaly detection using
    **Isolation Forest**.
    """
)


# ============================================================
# HISTORICAL STATISTICS
# ============================================================

total_readings = len(df)

anomaly_count = int(
    (df["prediction"] == 1).sum()
)

normal_count = int(
    (df["prediction"] == 0).sum()
)

anomaly_rate = (
    anomaly_count / total_readings
) * 100


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Readings",
        f"{total_readings:,}"
    )

with col2:
    st.metric(
        "Anomalies Detected",
        f"{anomaly_count:,}"
    )

with col3:
    st.metric(
        "Normal Readings",
        f"{normal_count:,}"
    )

with col4:
    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )


st.divider()

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("🧠 Model Performance")

st.write(
    "Evaluation results calculated on unseen test data."
)

perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

with perf_col1:
    st.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

with perf_col2:
    st.metric(
        "Precision",
        f"{metrics['precision'] * 100:.2f}%"
    )

with perf_col3:
    st.metric(
        "Recall",
        f"{metrics['recall'] * 100:.2f}%"
    )

with perf_col4:
    st.metric(
        "F1 Score",
        f"{metrics['f1_score'] * 100:.2f}%"
    )
# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("📊 Confusion Matrix")

cm = np.array(metrics["confusion_matrix"])

cm_col1, cm_col2 = st.columns([1, 1])

with cm_col1:

    st.write("Prediction results on unseen test data:")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Anomaly"],
        columns=["Predicted Normal", "Predicted Anomaly"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )


with cm_col2:

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted Normal", "Predicted Anomaly"],
            y=["Actual Normal", "Actual Anomaly"],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
            hoverongaps=False
        )
    )

    fig_cm.update_layout(
        title="AI Prediction Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class"
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

model_info_col1, model_info_col2, model_info_col3 = st.columns(3)

with model_info_col1:
    st.info("🤖 Algorithm: Isolation Forest")

with model_info_col2:
    st.info("📚 Learning: Unsupervised")

with model_info_col3:
    st.info("🧪 Evaluation: 80/20 Train-Test Split")


# ============================================================
# LIVE SENSOR MONITORING
# ============================================================

st.header("📡 Live Sensor Monitoring")

st.write(
    "Generate a new sensor reading and let the trained "
    "Isolation Forest model classify it."
)


# ------------------------------------------------------------
# Buttons
# ------------------------------------------------------------

button_col1, button_col2, button_col3 = st.columns(3)


with button_col1:

    generate_normal = st.button(
        "🟢 Generate Normal Reading",
        use_container_width=True
    )


with button_col2:

    generate_anomaly = st.button(
        "🔴 Simulate Anomaly",
        use_container_width=True
    )


with button_col3:

    clear_live = st.button(
        "🗑️ Clear Live Data",
        use_container_width=True
    )


# ============================================================
# CLEAR LIVE DATA
# ============================================================

if clear_live:

    st.session_state.live_readings = pd.DataFrame(
        columns=[
            "timestamp",
            "temperature",
            "vibration",
            "pressure",
            "humidity",
            "prediction"
        ]
    )

    st.rerun()


# ============================================================
# GENERATE NORMAL SENSOR READING
# ============================================================

if generate_normal:

    new_reading = {
        "temperature": np.random.normal(30, 3),
        "vibration": np.random.normal(0.30, 0.08),
        "pressure": np.random.normal(101, 2),
        "humidity": np.random.normal(50, 5)
    }

    sensor_values = np.array([
        [
            new_reading["temperature"],
            new_reading["vibration"],
            new_reading["pressure"],
            new_reading["humidity"]
        ]
    ])

    prediction = model.predict(sensor_values)[0]

    anomaly = 1 if prediction == -1 else 0

    new_row = pd.DataFrame(
        [{
            "timestamp": datetime.now(),
            "temperature": new_reading["temperature"],
            "vibration": new_reading["vibration"],
            "pressure": new_reading["pressure"],
            "humidity": new_reading["humidity"],
            "prediction": anomaly
        }]
    )

    st.session_state.live_readings = pd.concat(
        [
            st.session_state.live_readings,
            new_row
        ],
        ignore_index=True
    )


# ============================================================
# GENERATE ANOMALOUS SENSOR READING
# ============================================================

if generate_anomaly:

    new_reading = {
        "temperature": np.random.uniform(75, 100),
        "vibration": np.random.uniform(2.5, 6.0),
        "pressure": np.random.uniform(120, 140),
        "humidity": np.random.uniform(80, 100)
    }

    sensor_values = np.array([
        [
            new_reading["temperature"],
            new_reading["vibration"],
            new_reading["pressure"],
            new_reading["humidity"]
        ]
    ])

    prediction = model.predict(sensor_values)[0]

    anomaly = 1 if prediction == -1 else 0

    new_row = pd.DataFrame(
        [{
            "timestamp": datetime.now(),
            "temperature": new_reading["temperature"],
            "vibration": new_reading["vibration"],
            "pressure": new_reading["pressure"],
            "humidity": new_reading["humidity"],
            "prediction": anomaly
        }]
    )

    st.session_state.live_readings = pd.concat(
        [
            st.session_state.live_readings,
            new_row
        ],
        ignore_index=True
    )


# ============================================================
# SHOW LATEST LIVE READING
# ============================================================

if len(st.session_state.live_readings) > 0:

    latest = st.session_state.live_readings.iloc[-1]

    st.subheader("Latest Sensor Reading")

    live_col1, live_col2, live_col3, live_col4 = st.columns(4)

    with live_col1:
        st.metric(
            "🌡️ Temperature",
            f"{latest['temperature']:.2f} °C"
        )

    with live_col2:
        st.metric(
            "📳 Vibration",
            f"{latest['vibration']:.2f}"
        )

    with live_col3:
        st.metric(
            "💨 Pressure",
            f"{latest['pressure']:.2f}"
        )

    with live_col4:
        st.metric(
            "💧 Humidity",
            f"{latest['humidity']:.2f}%"
        )


    # --------------------------------------------------------
    # Prediction result
    # --------------------------------------------------------

    if latest["prediction"] == 1:

        st.error(
            "🔴 ANOMALY DETECTED — Abnormal sensor pattern!"
        )

    else:

        st.success(
            "🟢 NORMAL — Sensor readings are within "
            "the learned normal pattern."
        )


# ============================================================
# LIVE HISTORY
# ============================================================

if len(st.session_state.live_readings) > 0:

    st.subheader("📋 Live Reading History")

    st.dataframe(
        st.session_state.live_readings.tail(10),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# HISTORICAL SENSOR MONITORING
# ============================================================

st.divider()

st.header("📊 Historical Sensor Monitoring")

# ============================================================
# HISTORICAL ANOMALIES
# ============================================================

historical_anomalies = df[
    df["prediction"] == 1
].copy()


# ============================================================
# TEMPERATURE
# ============================================================

st.subheader("🌡️ Temperature Monitoring")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["temperature"],
        mode="lines",
        name="Temperature"
    )
)

fig.add_trace(
    go.Scatter(
        x=historical_anomalies["timestamp"],
        y=historical_anomalies["temperature"],
        mode="markers",
        name="AI Anomaly",
        marker=dict(
            size=9,
            symbol="x"
        )
    )
)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Temperature (°C)",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# VIBRATION
# ============================================================

st.subheader("📳 Vibration Monitoring")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["vibration"],
        mode="lines",
        name="Vibration"
    )
)

fig.add_trace(
    go.Scatter(
        x=historical_anomalies["timestamp"],
        y=historical_anomalies["vibration"],
        mode="markers",
        name="AI Anomaly",
        marker=dict(
            size=9,
            symbol="x"
        )
    )
)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Vibration",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# PRESSURE
# ============================================================

st.subheader("💨 Pressure Monitoring")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["pressure"],
        mode="lines",
        name="Pressure"
    )
)

fig.add_trace(
    go.Scatter(
        x=historical_anomalies["timestamp"],
        y=historical_anomalies["pressure"],
        mode="markers",
        name="AI Anomaly",
        marker=dict(
            size=9,
            symbol="x"
        )
    )
)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Pressure",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# HUMIDITY
# ============================================================

st.subheader("💧 Humidity Monitoring")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["humidity"],
        mode="lines",
        name="Humidity"
    )
)

fig.add_trace(
    go.Scatter(
        x=historical_anomalies["timestamp"],
        y=historical_anomalies["humidity"],
        mode="markers",
        name="AI Anomaly",
        marker=dict(
            size=9,
            symbol="x"
        )
    )
)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Humidity (%)",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# PREPARE ANOMALY DATA
# ============================================================

historical_anomalies = df[
    df["prediction"] == 1
].copy()


# ============================================================
# TEMPERATURE GRAPH
# ============================================================

st.subheader("🌡️ Temperature Monitoring")

fig_temperature = go.Figure()

# Normal/overall sensor line
fig_temperature.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["temperature"],
        mode="lines",
        name="Temperature"
    )
)

# Anomaly points
fig_temperature.add_trace(
    go.Scatter(
        x=historical_anomalies["timestamp"],
        y=historical_anomalies["temperature"],
        mode="markers",
        name="Anomaly",
        marker=dict(
            size=9,
            symbol="x"
        )
    )
)

fig_temperature.update_layout(
    xaxis_title="Time",
    yaxis_title="Temperature (°C)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_temperature,
    use_container_width=True
)


# ============================================================
# VIBRATION GRAPH
# ============================================================

st.subheader("📳 Vibration Monitoring")

fig_vibration = go.Figure()

fig_vibration.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["vibration"],
        mode="lines",
        name="Vibration"
    )
)

fig_vibration.add_trace(
    go.Scatter(
        x=historical_anomalies["timestamp"],
        y=historical_anomalies["vibration"],
        mode="markers",
        name="Anomaly",
        marker=dict(
            size=9,
            symbol="x"
        )
    )
)

fig_vibration.update_layout(
    xaxis_title="Time",
    yaxis_title="Vibration",
    hovermode="x unified"
)

st.plotly_chart(
    fig_vibration,
    use_container_width=True
)


# ============================================================
# PRESSURE GRAPH
# ============================================================

st.subheader("💨 Pressure Monitoring")

fig_pressure = go.Figure()

fig_pressure.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["pressure"],
        mode="lines",
        name="Pressure"
    )
)

fig_pressure.add_trace(
    go.Scatter(
        x=historical_anomalies["timestamp"],
        y=historical_anomalies["pressure"],
        mode="markers",
        name="Anomaly",
        marker=dict(
            size=9,
            symbol="x"
        )
    )
)

fig_pressure.update_layout(
    xaxis_title="Time",
    yaxis_title="Pressure",
    hovermode="x unified"
)

st.plotly_chart(
    fig_pressure,
    use_container_width=True
)


# ============================================================
# HUMIDITY GRAPH
# ============================================================

st.subheader("💧 Humidity Monitoring")

fig_humidity = go.Figure()

fig_humidity.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["humidity"],
        mode="lines",
        name="Humidity"
    )
)

fig_humidity.add_trace(
    go.Scatter(
        x=historical_anomalies["timestamp"],
        y=historical_anomalies["humidity"],
        mode="markers",
        name="Anomaly",
        marker=dict(
            size=9,
            symbol="x"
        )
    )
)

fig_humidity.update_layout(
    xaxis_title="Time",
    yaxis_title="Humidity (%)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_humidity,
    use_container_width=True
)