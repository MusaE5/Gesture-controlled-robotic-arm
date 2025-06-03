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
RIGHT_SERVO = 0   # row 0: controls “right” joint up/down
LEFT_SERVO  = 1   # row 1: controls “left” joint up/down
BIG_SERVO   = 2   # row 2: controls base rotation left/right

# === Initial Angles and Limits ===
# Start all three servos in a neutral (middle) position:
current_right_angle = 35   # midpoint of [0..70]
current_left_angle  = 90   # midpoint of [0..180]
current_big_angle   = 90   # midpoint of [20..160]

# Define each servo’s safe range:
RIGHT_MIN, RIGHT_MAX = 0, 70      # “up”   = 0°, “down”  = 70°
LEFT_MIN,  LEFT_MAX  = 0, 180     # “down” = 0°, “up”    = 180°
BIG_MIN,   BIG_MAX   = 20, 160    # “right”= 20°, “left”  = 160°

# Apply initial positions
kit.servo[RIGHT_SERVO].angle = current_right_angle
kit.servo[LEFT_SERVO].angle  = current_left_angle
kit.servo[BIG_SERVO].angle   = current_big_angle

# === Data Collection Parameters ===
SAMPLE_RATE = 0.0375   # 20 Hz → 0.75 s window total
WINDOW_SIZE = 20       # matches training (20 samples)


def collect_window():
    """
    Collect exactly WINDOW_SIZE readings from MPU6050,
    pausing SAMPLE_RATE seconds between each sample.
    Returns a NumPy array of shape (WINDOW_SIZE, 6):
        [Ax, Ay, Az, Gx, Gy, Gz] for each row.
    """
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
    """
    Given a (WINDOW_SIZE, 6) array, compute mean & std for each axis.
    Output: shape (1, 12) → [mean_Ax, std_Ax, mean_Ay, std_Ay, ..., mean_Gz, std_Gz]
    """
    features = []
    for i in range(6):
        axis_data = window[:, i]
        features.append(np.mean(axis_data))
        features.append(np.std(axis_data))
    return np.array(features).reshape(1, -1)


def move_vertical_servos(direction):
    """
    For an “up” or “down” gesture, move BOTH right & left servos
    in 20° steps, respecting each joint’s limits:
      • “up”   = left ↑ (angle +20), right ↑? Actually: right’s “up” = 0.
                 So left        → increase toward LEFT_MAX
                     right       → decrease toward RIGHT_MIN
      • “down” = left ↓ (angle –20), right ↓? Actually: right’s “down” = +20 toward RIGHT_MAX

    Edge-case logic:
      • “up”: only skip if BOTH left==LEFT_MAX AND right==RIGHT_MIN.
      • “down”: skip if BOTH left==LEFT_MIN AND right==RIGHT_MAX.
      Otherwise, move both by 20°, clamped.
    """
    global current_left_angle, current_right_angle

    if direction == "up":
        # Check if *both* joints have reached the topmost position:
        if current_left_angle == LEFT_MAX and current_right_angle == RIGHT_MIN:
            print("Both joints already at UP limit. Cannot move higher.")
            return

        # Move left joint UP if it isn’t at its max:
        if current_left_angle < LEFT_MAX:
            current_left_angle = min(current_left_angle + 40, LEFT_MAX)
            kit.servo[LEFT_SERVO].angle = current_left_angle
        else:
            print("Left joint already at UP limit; skipping left movement.")

        # Move right joint UP if it isn’t at its min:
        if current_right_angle > RIGHT_MIN:
            current_right_angle = max(current_right_angle - 40, RIGHT_MIN)
            kit.servo[RIGHT_SERVO].angle = current_right_angle
        else:
            print("Right joint already at UP limit; skipping right movement.")

        print(f"→ After UP command: left={current_left_angle}°, right={current_right_angle}°")

    elif direction == "down":
        # Check if *both* joints have reached the bottommost position:
        if current_left_angle == LEFT_MIN and current_right_angle == RIGHT_MAX:
            print("Both joints already at DOWN limit. Cannot move lower.")
            return

        # Move left joint DOWN if it isn’t at its min:
        if current_left_angle > LEFT_MIN:
            current_left_angle = max(current_left_angle - 40, LEFT_MIN)
            kit.servo[LEFT_SERVO].angle = current_left_angle
        else:
            print("Left joint already at DOWN limit; skipping left movement.")

        # Move right joint DOWN if it isn’t at its max:
        if current_right_angle < RIGHT_MAX:
            current_right_angle = min(current_right_angle + 40, RIGHT_MAX)
            kit.servo[RIGHT_SERVO].angle = current_right_angle
        else:
            print("Right joint already at DOWN limit; skipping right movement.")

        print(f"→ After DOWN command: left={current_left_angle}°, right={current_right_angle}°")

    else:
        print("move_vertical_servos() received invalid direction:", direction)


def move_big_servo(direction):
    """
    For “left” or “right” gestures, move the base rotation (BIG_SERVO)
    in 20° steps, clamped to [BIG_MIN..BIG_MAX]:
      • “left”:  increase angle toward BIG_MAX
      • “right”: decrease angle toward BIG_MIN
    """
    global current_big_angle

    if direction == "left":
        if current_big_angle < BIG_MAX:
            current_big_angle = min(current_big_angle + 40, BIG_MAX)
            kit.servo[BIG_SERVO].angle = current_big_angle
            print(f"Big servo moved LEFT → {current_big_angle}°")
        else:
            print("Big servo already at LEFT limit.")

    elif direction == "right":
        if current_big_angle > BIG_MIN:
            current_big_angle = max(current_big_angle - 40, BIG_MIN)
            kit.servo[BIG_SERVO].angle = current_big_angle
            print(f"Big servo moved RIGHT → {current_big_angle}°")
        else:
            print("Big servo already at RIGHT limit.")

    else:
        print("move_big_servo() received invalid direction:", direction)


def main():
    print("Live Gesture → Servo Control (press Ctrl+C to stop)\n")
    try:
        while True:
            # 1) Read WINDOW_SIZE samples
            window = collect_window()

            # 2) Extract 12 features (mean & std for each axis)
            features = extract_features(window)

            # 3) Predict one of: “up”, “down”, “left”, “right”
            prediction = model.predict(features)[0]
            print(f"Predicted Gesture: {prediction}")

            # 4) Act on the prediction:
            if prediction == "up" or prediction == "down":
                move_vertical_servos(prediction)
            elif prediction == "left" or prediction == "right":
                move_big_servo(prediction)
            else:
                print("Unknown gesture; no movement.")

    except KeyboardInterrupt:
        print("\nStopping live control. Goodbye.")


if __name__ == "__main__":
    main()
