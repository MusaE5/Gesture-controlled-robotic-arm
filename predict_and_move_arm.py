from mpu6050 import mpu6050
import time
import numpy as np
import joblib
from adafruit_servokit import ServoKit

# === Sensor and Model Setup ===
sensor = mpu6050(0x68)
model = joblib.load("model.pkl")
kit = ServoKit(channels=16)

# === Servo Channel Assignments ===
LEFT_SERVO = 0    # Row 0 controls up/down for left joint
BIG_SERVO = 2     # Row 2 controls left/right rotation

# === Initial Angles and Limits ===
# Start both servos in a neutral middle position
current_left_angle = 90
current_big_angle = 90

BIG_MIN, BIG_MAX = 20, 160      # Safe right/left angles for big servo

# Apply initial positions
kit.servo[LEFT_SERVO].angle = current_left_angle
kit.servo[BIG_SERVO].angle  = current_big_angle

# === Data Collection Parameters ===
SAMPLE_RATE = 0.05   # 20 Hz
WINDOW_SIZE = 40     # Matches training window size

def collect_window():
    "Collects WINDOW_SIZE samples of accel/gyro data at SAMPLE_RATE intervals."
    data = []
    while len(data) < WINDOW_SIZE:
        accel = sensor.get_accel_data()
        gyro  = sensor.get_gyro_data()
        sample = [
            accel['x'], accel['y'], accel['z'],
            gyro['x'],  gyro['y'],  gyro['z']
        ]
        data.append(sample)
        time.sleep(SAMPLE_RATE)
    return np.array(data)

def extract_features(window):
    "Extracts mean and std for each of the 6 axes from the window."
    features = []
    for i in range(6):
        axis_data = window[:, i]
        features.append(np.mean(axis_data))
        features.append(np.std(axis_data))
    return np.array(features).reshape(1, -1)

def move_left_servo(direction):
    """Moves the left servo up or down in 20° steps within [LEFT_MIN, LEFT_MAX]."""
    global current_left_angle
    if direction == "up":
        if current_left_angle < LEFT_MAX:
            current_left_angle = min(current_left_angle + 20, LEFT_MAX)
            kit.servo[LEFT_SERVO].angle = current_left_angle
            print(f"Left servo moved UP to {current_left_angle}°")
        else:
            print("Left servo already at UP limit.")
    elif direction == "down":
        if current_left_angle > LEFT_MIN:
            current_left_angle = max(current_left_angle - 20, LEFT_MIN)
            kit.servo[LEFT_SERVO].angle = current_left_angle
            print(f"Left servo moved DOWN to {current_left_angle}°")
        else:
            print("Left servo already at DOWN limit.")

def move_big_servo(direction):
    """Moves the big (rotation) servo left or right in 20° steps within [BIG_MIN, BIG_MAX]."""
    global current_big_angle
    if direction == "left":
        if current_big_angle < BIG_MAX:
            current_big_angle = min(current_big_angle + 20, BIG_MAX)
            kit.servo[BIG_SERVO].angle = current_big_angle
            print(f"Big servo moved LEFT to {current_big_angle}°")
        else:
            print("Big servo already at LEFT limit.")
    elif direction == "right":
        if current_big_angle > BIG_MIN:
            current_big_angle = max(current_big_angle - 20, BIG_MIN)
            kit.servo[BIG_SERVO].angle = current_big_angle
            print(f"Big servo moved RIGHT to {current_big_angle}°")
        else:
            print("Big servo already at RIGHT limit.")

def main():
    print("Live Gesture → Servo Control (press Ctrl+C to stop)")
    try:
        while True:
            # 1. Collect a window of sensor data
            window   = collect_window()
            features = extract_features(window)
            # 2. Predict gesture
            prediction = model.predict(features)[0]
            print(f"Predicted Gesture: {prediction}")

            # 3. Map prediction to servo movement
            if prediction == "up":
                move_left_servo("up")
            elif prediction == "down":
                move_left_servo("down")
            elif prediction == "left":
                move_big_servo("left")
            elif prediction == "right":
                move_big_servo("right")
            else:
                print("Unknown gesture; no movement.")

    except KeyboardInterrupt:
        print("\nStopping live control.")

if __name__ == "__main__":
    main()
