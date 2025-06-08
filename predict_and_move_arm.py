from mpu6050 import mpu6050
import time
import numpy as np
import joblib
from adafruit_servokit import ServoKit

# Set up sensor and import model
sensor = mpu6050(0x68)
model = joblib.load("model.pkl")
kit = ServoKit(channels=16)

# Assign servos
RIGHT_SERVO = 9   # right joint
LEFT_SERVO  = 15  # left joint
BIG_SERVO   = 4   # base rotation

# Initial positions
current_right_angle = 35
current_left_angle  = 90
current_big_angle   = 90

# Limits
RIGHT_MIN, RIGHT_MAX = 0, 70
LEFT_MIN,  LEFT_MAX  = 0, 180
BIG_MIN,   BIG_MAX   = 20, 160

kit.servo[RIGHT_SERVO].angle = current_right_angle
kit.servo[LEFT_SERVO].angle  = current_left_angle
kit.servo[BIG_SERVO].angle   = current_big_angle

# Faster sampling for snappier response
SAMPLE_RATE = 0.02   # ≈0.6 s per window
WINDOW_SIZE = 20

def collect_window():
    data = []
    while len(data) < WINDOW_SIZE:
        accel = sensor.get_accel_data()
        gyro  = sensor.get_gyro_data()
        data.append([accel['x'], accel['y'], accel['z'],
                     gyro['x'],  gyro['y'],  gyro['z']])
        time.sleep(SAMPLE_RATE)
    return np.array(data)

def extract_features(window):
    features = []
    for i in range(6):
        axis = window[:, i]
        features += [np.mean(axis), np.std(axis)]
    return np.array(features).reshape(1, -1)

def move_vertical_servos(direction):
    """ Smooth up/down: four 10° steps with tiny pauses """
    global current_left_angle, current_right_angle
    step, pause = 10, 0.05

    if direction == "up":
        for _ in range(4):
            if current_right_angle > RIGHT_MIN:
                current_right_angle = max(current_right_angle - step, RIGHT_MIN)
                kit.servo[RIGHT_SERVO].angle = current_right_angle
            if current_left_angle < LEFT_MAX:
                current_left_angle = min(current_left_angle + step, LEFT_MAX)
                kit.servo[LEFT_SERVO].angle = current_left_angle
            time.sleep(pause)

    elif direction == "down":
        for _ in range(4):
            if current_left_angle > LEFT_MIN:
                current_left_angle = max(current_left_angle - step, LEFT_MIN)
                kit.servo[LEFT_SERVO].angle = current_left_angle
            if current_right_angle < RIGHT_MAX:
                current_right_angle = min(current_right_angle + step, RIGHT_MAX)
                kit.servo[RIGHT_SERVO].angle = current_right_angle
            time.sleep(pause)

def move_big_servo(direction):
    """ Smooth left/right: four 10° steps """
    global current_big_angle
    step, pause = 10, 0.05

    if direction == "left":
        for _ in range(4):
            if current_big_angle < BIG_MAX:
                current_big_angle = min(current_big_angle + step, BIG_MAX)
                kit.servo[BIG_SERVO].angle = current_big_angle
            time.sleep(pause)

    elif direction == "right":
        for _ in range(4):
            if current_big_angle > BIG_MIN:
                current_big_angle = max(current_big_angle - step, BIG_MIN)
                kit.servo[BIG_SERVO].angle = current_big_angle
            time.sleep(pause)

def display_status(prediction):
    print("="*40)
    print(f"Gesture Detected: {prediction.upper()}")
    print(f"Left Joint : {current_left_angle}°")
    print(f"Right Joint: {current_right_angle}°")
    print(f"Base       : {current_big_angle}°")
    print("="*40 + "\n")

def main():
    print("Starting Gesture-Controlled Arm (Ctrl+C to stop)\n")
    try:
        while True:
            window    = collect_window()
            features  = extract_features(window)
            prediction = model.predict(features)[0]

            display_status(prediction)
            if prediction in ("up","down"):
                move_vertical_servos(prediction)
            else:
                move_big_servo(prediction)

    except KeyboardInterrupt:
        print("\n Demo stopped.")

if __name__ == "__main__":
    main()
