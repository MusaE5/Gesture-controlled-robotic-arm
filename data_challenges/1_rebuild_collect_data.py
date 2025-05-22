from mpu6050 import mpu6050
import time
import csv

sensor = mpu6050(0x68)   # set MPU6050 I2C location
FILE = "gesture_data.csv"  # CSV file that the samples will appended to
SAMPLE_RATE = 0.05   # Break up the 2 second timer into 2/0.05 (40) rows of samples
RECORD_SECONDS = 2  #Time to record 1 gesture

def main():
    print("Gesture data recorder\n")
    print("Click CTRL C to stop recording\n")
    while true:
        try:
            label = input("Enter gesture label (up,down,right,left)").strip().lower() 
            if label:
                record_gesture(label)

        except KeyboardInterrupt:
            print("\nRecording stopped")

def record_gesture(label):
    print(f"Recording gesture: {label} for {RECORD_SECONDS} seconds...")
    data=[]

    start_time = time.time() # beginning time (ie 3 seconds)

    while time.time()-start_time< RECORD_SECONDS:  # whilst the difference between current time and start time<2
        accel = sensor.get_accel_data 
        gyro = sensor.get_gyro_data
        sample = [accel['x'], accel['Y'], accel['Z'], gyro['x'],gyro['Y'],gyro['Z'], label]

        data.append(sample) #append sample to temporary data list
        time.sleep(SAMPLE_RATE) #wait 0.05 seconds before getting next sample

    with open(FILE, "a", newline="" ) as f: # save to csv file (append)
        writer = csv.writer(f) # create object to handle writing rows
        writer.writerows(data) # write all rows from previous loop at once into FILE


    print(f"Saved {len(data)} samples with label {label} ")







            