# PulseGuard-TinyML
PulseGuard-TinyML is an experimental HRV-based stress classification project using a lightweight Random Forest model to categorize stress levels as Healthy, Caution, or Danger. It explores real-time, on-device machine learning deployment on the ESP32 using TinyML.


# PulseGuard-TinyML

### HRV-Based Stress Classification Using Machine Learning and ESP32

PulseGuard-TinyML is a project focused on exploring how Heart Rate Variability (HRV) can be used to identify different stress levels using machine learning.

The idea is to train a lightweight Random Forest model that can classify HRV data into three categories: **Healthy, Caution, and Danger**. The long-term goal is to run this model directly on an ESP32, making it possible to perform predictions on an edge device without depending on cloud processing.

## How It Works

The project uses a heart-rate/HRV dataset from Kaggle. RR intervals (the time between consecutive heartbeats) are used to obtain relevant HRV features, which are then passed to a trained Random Forest classifier.

The development process involves:
1. Preparing and processing the HRV dataset.
2. Extracting useful features from RR intervals.
3. Training a Random Forest classification model.
4. Testing the model and evaluating its performance.
5. Exploring deployment on the ESP32 using TinyML.

## Technologies

- Python
- Scikit-learn
- Random Forest
- HRV and RR interval analysis
- ESP32
- TinyML

## Project Goals

- Understand the relationship between HRV patterns and stress classification.
- Build a lightweight model suitable for embedded systems.
- Explore running machine learning models on the ESP32.
- Learn about the limitations of deploying ML on resource-constrained hardware.

## Current Progress

The initial focus is on dataset processing, model development, and performance evaluation. ESP32 deployment and integration with live sensor data are part of the development goals.

## Future Work

- Deploy the trained model on the ESP32.
- Integrate a heart-rate sensor for live data collection.
- Improve model performance and reliability.
- Optimize memory usage and inference time.

## Disclaimer

This is an experimental learning project. Its output is not medically validated and should not be used for medical diagnosis.

---

*Built to explore the intersection of machine learning, embedded systems, and health-related data analysis.*
