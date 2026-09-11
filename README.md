# AI Sensor Anomaly Detection System

An AI-powered industrial sensor monitoring system that detects abnormal sensor patterns using Machine Learning and provides real-time visualization through a Streamlit dashboard.

## Project Overview

Industrial machines generate continuous sensor data such as temperature, vibration, pressure, and humidity. Abnormal patterns can indicate potential equipment problems.

This project uses an Isolation Forest machine learning algorithm to identify unusual sensor readings and classify them as normal or anomalous.

## Features

- Synthetic industrial sensor data generation
- Data preprocessing and feature preparation
- Unsupervised anomaly detection
- Isolation Forest machine learning model
- Model evaluation using accuracy, precision, recall and F1-score
- Confusion matrix evaluation
- Real-time sensor simulation
- Live anomaly detection
- Interactive Streamlit dashboard
- Historical sensor monitoring
- Prediction history

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- Streamlit

## Machine Learning Algorithm

### Isolation Forest

Isolation Forest is an unsupervised machine learning algorithm designed for anomaly detection.

The model learns patterns from sensor features:

- Temperature
- Vibration
- Pressure
- Humidity

It then identifies sensor readings that significantly differ from the learned normal pattern.

## System Architecture

```text
Sensor Data
     ↓
Data Generation
     ↓
Data Preprocessing
     ↓
Feature Preparation
     ↓
Isolation Forest
     ↓
Anomaly Prediction
     ↓
Model Evaluation
     ↓
Streamlit Dashboard
     ↓
Live Sensor Monitoring