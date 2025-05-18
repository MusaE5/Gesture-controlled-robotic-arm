# Gesture-Controlled Robotic Arm

This project uses an MPU6050 motion sensor and a trained machine learning model to classify hand gestures in real time, which are then used to control a robotic arm.

## Components
- Raspberry Pi
- MPU6050 (IMU sensor)
- 4-servo robotic arm
- Python (with scikit-learn, pandas, etc.)

## Repo Structure

- `collect_data.py` — Script to collect and label gesture data
- `ml_challenges/` — Step-by-step ML learning (loading, windowing, feature extraction, training)
- `train_model.py` — Final training script
- `predict_live.py` — Real-time gesture prediction (in progress)

##  How It Works
1. Record gestures using the sensor glove
2. Train a Random Forest model on extracted mean/std features
3. Run `predict_live.py` to classify gestures in real time
4. Use predicted gestures to control the robotic arm

## Future Improvements
- Servo integration for real-time gesture control
- Smoother gesture transitions
- Live dashboard visualization

##  Made by [MusaE5](https://github.com/MusaE5)
