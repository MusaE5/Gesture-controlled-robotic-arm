from mpu6050 import mpu6050
import time
import numpy as np
import joblib
from adafruit_servokit import ServoKit

# Set up sensor and import model
sensor = mpu6050(0x68)
model = joblib.load("model.pkl")
kit = ServoKit(channels=16)

# Assign servos to rows in the driver hat
RIGHT_SERVO = 0   # right joint up/down
LEFT_SERVO  = 1   # left joint up/down
BIG_SERVO   = 2   # base rotation left/right

# Initial positions (midpoints of their respective ranges)
current_right_angle = 35    
current_left_angle  = 90   
current_big_angle   = 90   

# Angle limits for each servo
RIGHT_MIN, RIGHT_MAX = 0, 70      
LEFT_MIN,  LEFT_MAX  = 0, 180     
BIG_MIN,   BIG_MAX   = 20, 160    

kit.servo[RIGHT_SERVO].angle = current_right_angle
kit.servo[LEFT_SERVO].angle  = current_left_angle
kit.servo[BIG_SERVO].angle   = current_big_angle

SAMPLE_RATE = 0.0375
WINDOW_SIZE = 20


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
    """
    Move both up/down servos based on gesture.
    “up”   → left += 40, right -= 40
    “down” → left -= 40, right += 40
    Clamps movement if any joint is at its limit.
    """
    global current_left_angle, current_right_angle

    if direction == "up":
        if current_left_angle == LEFT_MAX and current_right_angle == RIGHT_MIN:
            print("Both joints already at UP limit.")
            return

        if current_left_angle < LEFT_MAX:
            current_left_angle = min(current_left_angle + 40, LEFT_MAX)
            kit.servo[LEFT_SERVO].angle = current_left_angle
        else:
            print("Left already at UP limit.")

        if current_right_angle > RIGHT_MIN:
            current_right_angle = max(current_right_angle - 40, RIGHT_MIN)
            kit.servo[RIGHT_SERVO].angle = current_right_angle
        else:
            print("Right already at UP limit.")

        print(f"→ UP: left={current_left_angle}°, right={current_right_angle}°")

    elif direction == "down":
        if current_left_angle == LEFT_MIN and current_right_angle == RIGHT_MAX:
            print("Both joints already at DOWN limit.")
            return

        if current_left_angle > LEFT_MIN:
            current_left_angle = max(current_left_angle - 40, LEFT_MIN)
            kit.servo[LEFT_SERVO].angle = current_left_angle
        else:
            print("Left already at DOWN limit.")

        if current_right_angle < RIGHT_MAX:
            current_right_angle = min(current_right_angle + 40, RIGHT_MAX)
            kit.servo[RIGHT_SERVO].angle = current_right_angle
        else:
            print("Right already at DOWN limit.")

        print(f"→ DOWN: left={current_left_angle}°, right={current_right_angle}°")

    else:
        print("Invalid direction sent to vertical movement.")


def move_big_servo(direction):
    """
    Rotate base servo in 40° steps.
    “left”  → angle += 40
    “right” → angle -= 40
    """
    global current_big_angle

    if direction == "left":
        if current_big_angle < BIG_MAX:
            current_big_angle = min(current_big_angle + 40, BIG_MAX)
            kit.servo[BIG_SERVO].angle = current_big_angle
            print(f"Base moved LEFT → {current_big_angle}°")
        else:
            print("Base already at LEFT limit.")

    elif direction == "right":
        if current_big_angle > BIG_MIN:
            current_big_angle = max(current_big_angle - 40, BIG_MIN)
            kit.servo[BIG_SERVO].angle = current_big_angle
            print(f"Base moved RIGHT → {current_big_angle}°")
        else:
            print("Base already at RIGHT limit.")

    else:
        print("Invalid direction sent to base rotation.")


def main():
    print("Live Gesture → Servo Control (Ctrl+C to stop)\n")
    try:
        while True:
            window   = collect_window()
            features = extract_features(window)
            prediction = model.predict(features)[0]
            print(f"Gesture: {prediction}")

            if prediction in ["up", "down"]:
                move_vertical_servos(prediction)
            elif prediction in ["left", "right"]:
                move_big_servo(prediction)
            else:
                print("Unknown gesture. No movement.")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
