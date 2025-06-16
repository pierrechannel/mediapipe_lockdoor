import time
import threading

# Mock GPIO for non-Raspberry Pi systems
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    print("Running on Raspberry Pi - GPIO enabled")
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False
    print("Running on non-Raspberry Pi system - GPIO mocked")
    
    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = 1
        LOW = 0
        
        @staticmethod
        def setmode(mode):
            print(f"Mock GPIO: setmode({mode})")
        
        @staticmethod
        def setup(pin, mode):
            print(f"Mock GPIO: setup(pin={pin}, mode={mode})")
        
        @staticmethod
        def output(pin, value):
            state = "HIGH" if value else "LOW"
            print(f"Mock GPIO: output(pin={pin}, value={state}) - {'UNLOCKED' if value else 'LOCKED'}")
        
        @staticmethod
        def cleanup():
            print("Mock GPIO: cleanup()")
    
    GPIO = MockGPIO()

class DoorLock:
    def __init__(self, lock_pin=18):
        self.LOCK_PIN = lock_pin
        self.unlock_duration = 5
        self.last_unlock_time = 0
        
        # GPIO setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.LOCK_PIN, GPIO.OUT)
        GPIO.output(self.LOCK_PIN, GPIO.LOW)
    
    def unlock(self):
        """Unlock the door"""
        current_time = time.time()
        
        # Prevent rapid unlocking
        if current_time - self.last_unlock_time < 2:
            return
        
        # Activate relay (unlock door)
        GPIO.output(self.LOCK_PIN, GPIO.HIGH)
        self.last_unlock_time = current_time
        
        # Schedule lock after duration
        def lock_door():
            time.sleep(self.unlock_duration)
            GPIO.output(self.LOCK_PIN, GPIO.LOW)
            print("Door locked automatically")
        
        threading.Thread(target=lock_door, daemon=True).start()
    
    def cleanup(self):
        """Clean up GPIO resources"""
        GPIO.output(self.LOCK_PIN, GPIO.LOW)
        GPIO.cleanup()