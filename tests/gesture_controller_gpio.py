import time
import numpy as np
import joblib
from mpu6050 import mpu6050
import RPi.GPIO as GPIO

# Servo set up
SERVO_PIN = 18  # BCM pin 18 (physical pin 12)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz
pwm.start(7.5)  # center position

#Load Model and sensor
model = joblib.load("model.pkl")
sensor = mpu6050(0x68)

# Map gesture to duty cycles
GESTURE_TO_DUTY = {
    "left": 5.0,    # ~0°
    "right": 10.0,  # ~180°
    "up": 7.5,      # ~90°
    "down": 6.2     # ~45°
}

def collect_window(window_size=40, sample_rate=0.05):
    data = []
    while len(data) < window_size:
        accel = sensor.get_accel_data()
        gyro = sensor.get_gyro_data()
        data.append([accel['x'], accel['y'], accel['z'],
                     gyro['x'], gyro['y'], gyro['z']])
        time.sleep(sample_rate)
    return np.array(data)

def extract_features(window):
    feats = []
    for i in range(6):
        axis = window[:, i]
        feats.append(np.mean(axis))
        feats.append(np.std(axis))
    return np.array(feats).reshape(1, -1)

try:
    print("Starting gesture→servo control; Ctrl+C to quit")
    while True:
        print("Collecting gesture...")
        w = collect_window()
        features = extract_features(w)
        gesture = model.predict(features)[0]
        print(f"Predicted gesture: {gesture}")

        duty = GESTURE_TO_DUTY.get(gesture, 7.5)
        pwm.ChangeDutyCycle(duty)
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping.")

finally:
    pwm.stop()
    GPIO.cleanup()
