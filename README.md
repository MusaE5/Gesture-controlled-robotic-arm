# Gesture-Controlled Robotic Arm

This project uses an MPU6050 motion sensor and a trained machine learning model to classify hand gestures in real time, which are then used to control a robotic arm.

---

## Components

- Raspberry Pi
- MPU6050 (IMU sensor)
- 4-servo robotic arm
- Python 3 (scikit-learn, pandas, numpy, joblib)

---

## Repository Structure

- `collect_data.py` — Records labeled gesture data from the glove
- `train_model.py` — Final training script using Random Forest
- `predict_live.py` — Real-time gesture classification using live sensor input
- `ml_challenges/` — Learning scripts that break down model training step by step:
  - `1_load_csv.py` — Load and view gesture data
  - `2_windowing.py` — Slice data into 2-second windows
  - `3_extract_features.py` — Extract mean/std features from each window
  - `4_train_model.py` — End-to-end training and evaluation pipeline
  - `check_label_counts.py` — Utility script to debug class imbalance in training data

---

## How It Works

1. Record labeled gestures using the MPU6050 glove with `collect_data.py`
2. Extract features and train a Random Forest model using `train_model.py`
3. Use `predict_live.py` to classify gestures from live data
4. Predicted gestures will soon be used to control a robotic arm in real time

---

## Debugging and Model Improvements

- Initially, the model consistently misclassified gestures like "up" and "down" as "left"
- Using `check_label_counts.py`, I discovered a class imbalance in the training windows
- After recording additional samples and retraining, live predictions became highly accurate and consistent

---

## Future Improvements

- Servo integration for gesture-based arm control
- Confidence thresholds to filter out uncertain predictions
- Live visualization dashboard to monitor real-time predictions
- Feature engineering: extract additional statistical features such as range, max-min difference,   acceleration magnitude, or temporal patterns to improve gesture separability
- Evaluate and compare alternative classifiers (e.g. KNN, SVM, or neural networks) for improved generalization

---

## Author

Made by [MusaE5](https://github.com/MusaE5)