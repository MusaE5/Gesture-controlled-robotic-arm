# Gesture-Controlled Robotic Arm

This project uses an MPU6050 motion sensor and a trained machine learning model to classify hand gestures in real time, which are then used to control a 4-DOF robotic arm. Built with Python on a Raspberry Pi, it combines embedded systems, real-time ML inference, and sensor-based control.

---

##  How It Works
1. Record labeled gestures using `collect_data.py`
2. Train a Random Forest classifier on mean/std features using `train_model.py`
3. Use `predict_and_move_arm.py` to detect real-time gestures and move the robotic arm

This demo system is optimized for responsiveness and visual clarity, using a reduced window size and large servo steps.

---

##  Key Features
- **Live Gesture Classification**: Predicts from 4 gestures: up, down, left, right
- **Real-Time Servo Control**: Servo angles update instantly with predictions
- **Optimized Windowing**: Gesture window size reduced from 40 to 20, improving responsiveness to ~0.75s per gesture
- **Modular Design**: Scripts separated by purpose (collection, training, prediction, testing)
- **Learning-Focused Folders**: Documented breakdown of the ML and data pipeline

---

## 📁 Repository Structure

| File/Folder            | Description |
|------------------------|-------------|
| `collect_data.py`      | Record labeled gesture data from the MPU6050 glove |
| `train_model.py`       | Extract features, train Random Forest, save `model.pkl` |
| `predict_and_move_arm.py` | Main script for real-time gesture prediction and servo motion |
| `tests/predict_live.py`| Old standalone prediction script (archived) |
| `tests/test_servo_single.py` | Servo diagnostic test to verify wiring and movement |
| `ml_challenges/`       | ML pipeline breakdown: load, window, extract, train |
| `data_challenges/`     | Rebuilt/experimental gesture data collection variants |

---

##  Learning and Understanding (Challenge Folders)

### `ml_challenges/`
Rebuilt the full model pipeline step-by-step for deeper understanding:
- `1_load_csv.py` — Load and explore gesture dataset
- `2_windowing.py` — Slice time-series data into gesture windows
- `3_extract_features.py` — Extract mean/std for each axis
- `4_train_model.py` — Full pipeline for training and evaluating classifier

### `data_challenges/`
Custom data capture experiments for validation and improved control:
- `rebuild_collect_data.py` — Manual sampling + comments for learning


---

## 🛠 Debugging & Model Evolution
- Early version misclassified "up/down" as "left/right" due to class imbalance
- Balanced the dataset after running `check_label_counts.py`
- Retrained model yielded **~94% accuracy** even after reducing window size
- Collected more diverse samples and shortened gesture duration to improve responsiveness

---

##  Window Size Optimization
- **Original**: WINDOW_SIZE = 40 (~1.5–2s delay)
- **Optimized**: WINDOW_SIZE = 20 (~0.75s per prediction)
- Maintained accuracy, reduced latency
- Greatly improved live feel and responsiveness of the robotic arm

---

##  Future Improvements
- Smooth servo motion with easing transitions
- Add prediction confidence filtering to reduce noise
- Use magnitude, derivative, or axis combinations for better feature separation
- Try additional classifiers (KNN, SVM, lightweight neural networks)
- Streamlit or Matplotlib dashboard to visualize live prediction/angle updates

---

## 🎥 Demo Video
[Gesture-Controlled Robotic Arm — YouTube Demo](https://youtu.be/0qojFLV_fl0)

Demonstrates gesture prediction and servo movement in real time (left, right, up, down). Video includes project overview, voiceover, and synchronized terminal + hardware visuals.

---

##  Author
Made by **MusaE5** — .

---

##  Notes
- Minimal comments are used in production scripts to maintain clarity
- Fully commented learning-focused scripts are in `ml_challenges/` and `data_challenges/`
- `predict_and_move_arm.py` is the **final live system** used in the video
- `predict_live.py` has been archived to `tests/` to reflect repository cleanup
