# Gesture-Controlled Robotic Arm

This project uses an MPU6050 motion sensor and a trained machine learning model to classify hand gestures in real time, which are then used to control a 4-DOF robotic arm. It's built with Python on a Raspberry Pi and combines embedded systems, real-time ML inference, and sensor-based control.

---

## Components

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

## 📁 Repository Structure

| File/Folder                  | Description |
|------------------------------|-------------|
| `collect_data.py`            | Record labeled gesture data from the MPU6050 glove |
| `train_model.py`             | Final script to extract features, train Random Forest, and save `model.pkl` |
| `predict_live.py`            | Real-time gesture prediction using live sensor input |
| `move_servo_predict.py`      | Uses live gesture predictions to move servos (row 0 for up/down, row 2 for left/right) |
| `check_label_counts.py`      | Check for gesture imbalance in training data |
| `tests/test_servo_single.py` | Diagnostic script to verify PCA9685 wiring and observe servo behavior |
| `tests/`                     | Temporary/test scripts for debugging and validation |

### Learning and Understanding (Not Used in Final Pipeline)

| Folder             | Purpose |
|--------------------|---------|
| `ml_challenges/`   | Breakdown of full model pipeline into step-by-step learning scripts:<br>  - `1_load_csv.py` — Load and explore dataset  <br>  - `2_windowing.py` — Slice data into time-based windows  <br>  - `3_extract_features.py` — Extract mean/std features per axis  <br>  - `4_train_model.py` — End-to-end train/test split and evaluation |
| `data_challenges/` | Rebuilt versions of gesture data collection scripts for deeper understanding and validation (e.g., custom timed sampling, accelerometer-only capture) |

> These challenge folders were used to **rebuild the full system from scratch** and reflect my effort to learn machine learning and embedded systems deeply — not just run code.

---

## How It Works

1. Record labeled gestures using `collect_data.py`
2. Train a gesture classification model with `train_model.py` using mean/std features
3. Use `predict_live.py` to classify real-time gestures
4. Use `move_servo_predict.py` to control robotic arm movement based on predicted gestures

---

## Debugging Process & Model Evolution

- The initial model misclassified gestures like `"up"` and `"down"` as `"left"` due to class imbalance
- Used `check_label_counts.py` to identify the imbalance and collected more samples
- Retrained the model with balanced data and achieved highly reliable predictions
- Built additional challenge scripts to re-understand each phase of the pipeline (data, model, prediction)
 Window Size Optimization
To improve responsiveness, the model was retrained using a reduced WINDOW_SIZE = 20 (from the original 40), cutting gesture recognition time from ~2 seconds to ~0.75 seconds.
Despite the shorter window, classification accuracy remained high (94%) with minimal misclassification.
This change significantly improved the real-time feel of the robotic arm.

---

## Future Improvements

- Smooth servo motion using gradual angle changes and bounded movement ranges
- Add prediction confidence thresholds to reduce false positives
- Reduce gesture collection time to improve responsiveness (e.g. 1 second)
- Streamlit/Matplotlib live dashboard to visualize predictions
- Add features: magnitude, range, or derivatives to improve separability
- Evaluate classifiers beyond Random Forest (e.g. KNN, SVM, shallow neural nets)

---

## Demo

- [Live gesture → servo reaction demo (YouTube)]https://www.youtube.com/shorts/UtrVyAoeRxM 
- Currently shows left/right motion using real-time prediction and servo control
- More complete videos (including full robotic arm motion) to be added soon

---

## Author

Made by [MusaE5](https://github.com/MusaE5)