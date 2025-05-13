from mpu6050 import mpu6050
import time
import csv

sensor = mpu6050(0x68)
FILENAME = "gesture_data.csv"
RECORD_SECONDS = 2
SAMPLE_RATE = 0.05  # ~20 samples/sec

def record_gesture(label):
    print(f"\nRecording gesture: {label} for {RECORD_SECONDS} seconds...")
    data = []

    start_time = time.time()
    while time.time() - start_time < RECORD_SECONDS:
        accel = sensor.get_accel_data()
        gyro = sensor.get_gyro_data()
        sample = [
            accel['x'], accel['y'], accel['z'],
            gyro['x'], gyro['y'], gyro['z'],
            label
        ]
        data.append(sample)
        time.sleep(SAMPLE_RATE)

    # Save to CSV
    with open(FILENAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"Saved {len(data)} samples with label: {label}")

def main():
    print("Gesture Data Recorder")
    print("Press Ctrl+C to stop.")
    while True:
        try:
            label = input("\nEnter gesture label (e.g. up, down, left): ").strip().lower()
            if label:
                record_gesture(label)
        except KeyboardInterrupt:
            print("\nRecording stopped.")
            break

if __name__ == "__main__":
    main()
