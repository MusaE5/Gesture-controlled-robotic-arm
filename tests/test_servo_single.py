from adafruit_servokit import ServoKit
import time

# Initialize the PCA9685 with 16 channels
kit = ServoKit(channels=16)

# Set the servo range just in case (usually 0 to 180 degrees is standard)
kit.servo[0].set_pulse_width_range(500, 2500)

print("=== Servo Angle Test ===")
print("Type an angle between 0 and 180 to move the servo on channel 0.")
print("Press Ctrl+C to exit.\n")

try:
    while True:
        angle_str = input("Enter angle (0–180): ").strip()
        if angle_str.isdigit():
            angle = int(angle_str)
            if 0 <= angle <= 180:
                kit.servo[0].angle = angle
                print(f"Moved to {angle}°")
            else:
                print("Angle must be between 0 and 180.")
        else:
            print(" Invalid input. Please enter a number.")
        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nTest ended.")

