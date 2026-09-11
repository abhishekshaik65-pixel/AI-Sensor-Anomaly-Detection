import pandas as pd

# Sensor features used by the ML model
FEATURES = [
    "temperature",
    "vibration",
    "pressure",
    "humidity"
]


def load_sensor_data(file_path="data/sensor_data.csv"):
    """
    Load sensor data from CSV file.
    """
    df = pd.read_csv(file_path)

    return df


def prepare_features(df):
    """
    Prepare sensor features for machine learning.
    """

    # Select only sensor columns
    X = df[FEATURES].copy()

    # Check for missing values
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())

    return X


if __name__ == "__main__":

    print("=" * 50)
    print("SENSOR DATA PREPROCESSING")
    print("=" * 50)

    # Load dataset
    df = load_sensor_data()

    print(f"\nDataset shape: {df.shape}")

    # Prepare features
    X = prepare_features(df)

    print("\nFeatures used by the ML model:")
    print(FEATURES)

    print("\nFeature data preview:")
    print(X.head())

    print("\nMissing values:")
    print(X.isnull().sum())

    print("\nPreprocessing completed successfully!")