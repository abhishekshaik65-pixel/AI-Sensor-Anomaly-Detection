import os
import json
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessing import load_sensor_data, prepare_features


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("AI SENSOR ANOMALY DETECTION - MODEL TRAINING")
print("=" * 60)

df = load_sensor_data()

print(f"\nDataset loaded successfully!")
print(f"Total readings: {len(df)}")


# ============================================================
# 2. PREPARE FEATURES
# ============================================================

X = prepare_features(df)

# Used only for evaluation.
# Isolation Forest does NOT use these labels for training.
y = df["true_anomaly"]

print("\nFeatures used for training:")
print(list(X.columns))


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset split:")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")


# ============================================================
# 4. CREATE MODEL
# ============================================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\nTraining Isolation Forest model...")

model.fit(X_train)

print("Model training completed!")


# ============================================================
# 6. PREDICT TEST DATA
# ============================================================

print("\nPredicting unseen test data...")

raw_predictions = model.predict(X_test)

# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

y_pred = (raw_predictions == -1).astype(int)


# ============================================================
# 7. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 9. DISPLAY EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION ON UNSEEN TEST DATA")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)


# ============================================================
# 10. SAVE METRICS
# ============================================================

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist()
}

os.makedirs(
    "model",
    exist_ok=True
)

with open(
    "model/metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

print("\nModel metrics saved to: model/metrics.json")


# ============================================================
# 11. TRAIN FINAL MODEL ON COMPLETE DATASET
# ============================================================

print("\nTraining final model using complete dataset...")

final_model = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X)

print("Final model training completed!")


# ============================================================
# 12. GENERATE FULL DATASET PREDICTIONS
# ============================================================

full_predictions = final_model.predict(X)

df["prediction"] = (
    full_predictions == -1
).astype(int)


# ============================================================
# 13. SAVE PREDICTIONS
# ============================================================

output_file = "data/sensor_predictions.csv"

df.to_csv(
    output_file,
    index=False
)

print(
    f"\nPredictions saved to: {output_file}"
)


# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

model_path = "model/anomaly_model.pkl"

joblib.dump(
    final_model,
    model_path
)

print(
    f"Trained model saved to: {model_path}"
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

normal_count = int(
    (df["prediction"] == 0).sum()
)

anomaly_count = int(
    (df["prediction"] == 1).sum()
)

print("\n" + "=" * 60)
print("FINAL PREDICTION SUMMARY")
print("=" * 60)

print(
    f"\nNormal readings detected : {normal_count}"
)

print(
    f"Anomalies detected       : {anomaly_count}"
)


print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)