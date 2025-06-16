import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)  # Use Broadcom numbering
GPIO.setup(18, GPIO.OUT)

try:
    while True:
        GPIO.output(18, GPIO.HIGH)  # Turn on
        print("GPIO 18 ON")
        time.sleep(1)
        GPIO.output(18, GPIO.LOW)   # Turn off
        print("GPIO 18 OFF")
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()