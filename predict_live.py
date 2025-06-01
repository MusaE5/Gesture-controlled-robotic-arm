from mpu6050 import mpu6050
import time
import numpy as np
import joblib

# Initialize sensor and model
sensor = mpu6050(0x68)
model = joblib.load("model.pkl")

SAMPLE_RATE = 0.0375  
WINDOW_SIZE = 20    

def collect_window():
    print("\nCollecting gesture window...")
    data = []

    while len(data) < WINDOW_SIZE:
        accel = sensor.get_accel_data()
        gyro = sensor.get_gyro_data()

        sample = [
            accel['x'], accel['y'], accel['z'],
            gyro['x'], gyro['y'], gyro['z']
        ]
        data.append(sample)
        time.sleep(SAMPLE_RATE)

    print(f"Collected {len(data)} samples.")
    return np.array(data)

def extract_features(window):
    feature_vector = []
    for i in range(6):  # 6 sensor axes
        axis_data = window[:, i]
        feature_vector.append(np.mean(axis_data))
        feature_vector.append(np.std(axis_data))
    return np.array(feature_vector).reshape(1, -1)

def main():
    print("Live Gesture Prediction (press Ctrl+C to stop)")

    try:
        while True:
            window = collect_window()
            features = extract_features(window)
            prediction = model.predict(features)[0]
            print(f"Predicted Gesture: {prediction}")

    except KeyboardInterrupt:
        print("\nPrediction stopped.")

if __name__ == "__main__":
    main()
