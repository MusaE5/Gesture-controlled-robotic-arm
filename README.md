# Gesture-Controlled Robotic Arm

This project uses an MPU6050 motion sensor and a trained machine learning model to classify hand gestures in real time, which are then used to control a 4-DOF robotic arm. It's built with Python on a Raspberry Pi and combines embedded systems, real-time ML inference, and sensor-based control.

---

##  Components

- Raspberry Pi (any model with I2C support)
- MPU6050 (accelerometer + gyroscope)
- MG90S servos (x4)
- PCA9685 servo driver (via I2C)
- Python 3 with:
  - scikit-learn
  - pandas
  - numpy
  - joblib

---

##  Repository Structure

| File/Folder             | Description |
|-------------------------|-------------|
| `collect_data.py`       | Record labeled gesture data from the MPU6050 glove |
| `train_model.py`        | Final script to extract features, train Random Forest, and save `model.pkl` |
| `predict_live.py`       | Real-time gesture prediction using live sensor input |
| `gesture_controller_gpio.py` | Single-servo gesture test using GPIO (archived in `tests/`) |
| `tests/`                | Temporary/test scripts (e.g., GPIO-based servo controller) |
| `ml_challenges/`        | Learning scripts that break down ML pipeline:
  - `1_load_csv.py` — Load and explore dataset
  - `2_windowing.py` — Slice data into time-based windows
  - `3_extract_features.py` — Extract mean/std features per axis
  - `4_train_model.py` — End-to-end train/test split and evaluation
| `check_label_counts.py` | Check for gesture imbalance in training data |

---

##  How It Works

1. Record labeled gestures using `collect_data.py`
2. Train a gesture classification model with `train_model.py` using mean/std features
3. Use `predict_live.py` to classify real-time gestures
4. (Coming soon) Map classified gestures to servo movements to control a 4-DOF robotic arm

---

##  Debugging Process & Model Evolution

- The initial model misclassified gestures like `"up"` and `"down"` as `"left"` due to class imbalance
- After using `check_label_counts.py`, I recorded additional training data
- Retraining the model improved accuracy significantly — `predict_live.py` now makes highly reliable predictions across all 4 classes

---

##  Future Improvements

-  **Servo integration** via PCA9685 to control a full robotic arm (arriving soon)
-  Add **prediction confidence thresholds** to reduce false positives
-  **Live visualization dashboard** (Streamlit or Matplotlib) for real-time predictions
-  Feature engineering: add `range`, `magnitude`, or `temporal derivatives`
-  Evaluate classifiers beyond Random Forest (KNN, SVM, shallow MLPs)

---

##  Demo (Coming Soon)

> To be added once full servo control is working:

-  Terminal screen recording of live predictions
-  Video showing servo reacting to hand gestures
-  Final demo: glove → full robotic arm motion via ML pipeline

---

##  Author

Made by [MusaE5](https://github.com/MusaE5)

---

