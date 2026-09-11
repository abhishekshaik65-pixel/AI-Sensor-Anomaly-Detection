import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# -----------------------------
# Configuration
# -----------------------------

NUM_SAMPLES = 5000
ANOMALY_PERCENTAGE = 0.03

np.random.seed(42)

# -----------------------------
# Generate timestamps
# -----------------------------

start_time = datetime.now()

timestamps = [
    start_time + timedelta(seconds=i)
    for i in range(NUM_SAMPLES)
]

# -----------------------------
# Generate normal sensor data
# -----------------------------

temperature = np.random.normal(
    loc=30,
    scale=3,
    size=NUM_SAMPLES
)

vibration = np.random.normal(
    loc=0.30,
    scale=0.08,
    size=NUM_SAMPLES
)

pressure = np.random.normal(
    loc=101,
    scale=2,
    size=NUM_SAMPLES
)

humidity = np.random.normal(
    loc=50,
    scale=5,
    size=NUM_SAMPLES
)

# -----------------------------
# Create anomaly labels
# -----------------------------

true_anomaly = np.zeros(NUM_SAMPLES, dtype=int)

num_anomalies = int(NUM_SAMPLES * ANOMALY_PERCENTAGE)

anomaly_indices = np.random.choice(
    NUM_SAMPLES,
    num_anomalies,
    replace=False
)

# -----------------------------
# Inject abnormal sensor values
# -----------------------------

for index in anomaly_indices:

    temperature[index] = np.random.uniform(75, 100)

    vibration[index] = np.random.uniform(2.5, 6.0)

    pressure[index] = np.random.uniform(120, 140)

    humidity[index] = np.random.uniform(80, 100)

    true_anomaly[index] = 1

# -----------------------------
# Create DataFrame
# -----------------------------

df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "humidity": humidity,
    "true_anomaly": true_anomaly
})

# -----------------------------
# Create data directory
# -----------------------------

os.makedirs("data", exist_ok=True)

# -----------------------------
# Save dataset
# -----------------------------

output_path = "data/sensor_data.csv"

df.to_csv(output_path, index=False)

# -----------------------------
# Display information
# -----------------------------

print("=" * 50)
print("SENSOR DATA GENERATION COMPLETED")
print("=" * 50)

print(f"Total readings      : {len(df)}")
print(f"Normal readings     : {(true_anomaly == 0).sum()}")
print(f"Anomalies generated : {(true_anomaly == 1).sum()}")

print("\nDataset preview:")
print(df.head())

print(f"\nDataset saved to: {output_path}")