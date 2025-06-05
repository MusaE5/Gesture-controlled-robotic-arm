from mpu6050 import mpu6050
import time
import numpy as np
import joblib
from adafruit_servokit import ServoKit
import csv
from datetime import datetime

# Set up sensor and import model
sensor = mpu6050(0x68)
model = joblib.load("model.pkl")
kit = ServoKit(channels=16)

# Assign servos to rows in the driver hat
RIGHT_SERVO = 0   # right joint up/down
LEFT_SERVO  = 1   # left joint up/down
BIG_SERVO   = 2   # base rotation left/right

# Initial positions
current_right_angle = 35    
current_left_angle  = 90   
current_big_angle   = 90   

# CSV Logging Setup
log_file = open("gesture_log.csv", mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["timestamp", "gesture", "left_angle", "right_angle", "base_angle"])

# Angle limits
RIGHT_MIN, RIGHT_MAX = 0, 70      
LEFT_MIN,  LEFT_MAX  = 0, 180     
BIG_MIN,   BIG_MAX   = 20, 160    

kit.servo[RIGHT_SERVO].angle = current_right_angle
kit.servo[LEFT_SERVO].angle  = current_left_angle
kit.servo[BIG_SERVO].angle   = current_big_angle

SAMPLE_RATE = 0.0375
WINDOW_SIZE = 20

def log_prediction(prediction):
    """Log prediction + angles to CSV and print to terminal."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    csv_writer.writerow([timestamp, prediction, current_left_angle, current_right_angle, current_big_angle])
    log_file.flush()

    print(f"[{timestamp}] Gesture: {prediction.upper()} | Left: {current_left_angle}° | Right: {current_right_angle}° | Base: {current_big_angle}°")

def collect_window():
    """Collect 20 readings of [Ax, Ay, Az, Gx, Gy, Gz] from the MPU6050."""
    data = []
    while len(data) < WINDOW_SIZE:
        accel = sensor.get_accel_data()
        gyro  = sensor.get_gyro_data()
        data.append([
            accel['x'], accel['y'], accel['z'],
            gyro['x'],  gyro['y'],  gyro['z']
        ])
        time.sleep(SAMPLE_RATE)
    return np.array(data)

def extract_features(window):
    """Return mean & std dev for each axis → shape: (1, 12)."""
    features = []
    for i in range(6):
        axis_data = window[:, i]
        features.append(np.mean(axis_data))
        features.append(np.std(axis_data))
    return np.array(features).reshape(1, -1)

def move_vertical_servos(direction):
    global current_left_angle, current_right_angle

    if direction == "up":
        if current_left_angle == LEFT_MAX and current_right_angle == RIGHT_MIN:
            return

        if current_left_angle < LEFT_MAX:
            current_left_angle = min(current_left_angle + 40, LEFT_MAX)
            kit.servo[LEFT_SERVO].angle = current_left_angle

        if current_right_angle > RIGHT_MIN:
            current_right_angle = max(current_right_angle - 40, RIGHT_MIN)
            kit.servo[RIGHT_SERVO].angle = current_right_angle

    elif direction == "down":
        if current_left_angle == LEFT_MIN and current_right_angle == RIGHT_MAX:
            return

        if current_left_angle > LEFT_MIN:
            current_left_angle = max(current_left_angle - 40, LEFT_MIN)
            kit.servo[LEFT_SERVO].angle = current_left_angle

        if current_right_angle < RIGHT_MAX:
            current_right_angle = min(current_right_angle + 40, RIGHT_MAX)
            kit.servo[RIGHT_SERVO].angle = current_right_angle

def move_big_servo(direction):
    global current_big_angle

    if direction == "left":
        if current_big_angle < BIG_MAX:
            current_big_angle = min(current_big_angle + 40, BIG_MAX)
            kit.servo[BIG_SERVO].angle = current_big_angle

    elif direction == "right":
        if current_big_angle > BIG_MIN:
            current_big_angle = max(current_big_angle - 40, BIG_MIN)
            kit.servo[BIG_SERVO].angle = current_big_angle

def main():
    print("Live Gesture → Servo Control (Ctrl+C to stop)\n")
    try:
        while True:
            window   = collect_window()
            features = extract_features(window)
            prediction = model.predict(features)[0]
            log_prediction(prediction)

            if prediction in ["up", "down"]:
                move_vertical_servos(prediction)
            elif prediction in ["left", "right"]:
                move_big_servo(prediction)

    except KeyboardInterrupt:
        print("\nStopped.")
        log_file.close()

if __name__ == "__main__":
    main()
