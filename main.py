import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from datetime import datetime

# Mock GPIO for non-Raspberry Pi systems
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    print("Running on Raspberry Pi - GPIO enabled")
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False
    print("Running on non-Raspberry Pi system - GPIO mocked")
    
    # Create a mock GPIO module
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

class FaceDoorLock:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Face detection and mesh
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # GPIO setup for door lock (relay control)
        self.LOCK_PIN = 18  # GPIO pin for relay
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.LOCK_PIN, GPIO.OUT)
        GPIO.output(self.LOCK_PIN, GPIO.LOW)  # Lock initially
        
        # Face database
        self.known_faces = {}
        self.face_db_file = "face_database.pkl"
        self.load_face_database()
        
        # Settings
        self.recognition_threshold = 0.6  # Similarity threshold
        self.unlock_duration = 5  # Seconds to keep door unlocked
        self.last_unlock_time = 0
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        gpio_status = "enabled" if GPIO_AVAILABLE else "mocked"
        print(f"Face Recognition Door Lock System Initialized (GPIO {gpio_status})")
        print("Commands: 'r' - Register face, 'q' - Quit, 's' - Show database, 'd' - Delete face")

    def extract_face_features(self, image, face_landmarks):
        """Extract facial features from MediaPipe landmarks"""
        h, w = image.shape[:2]
        
        # Key facial landmarks indices (68-point model equivalent)
        key_points = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        # Extract coordinates
        features = []
        for idx in key_points:
            if idx < len(face_landmarks.landmark):
                x = face_landmarks.landmark[idx].x * w
                y = face_landmarks.landmark[idx].y * h
                features.extend([x, y])
        
        # Calculate relative distances and angles
        if len(features) >= 4:
            # Normalize features relative to face size
            face_width = abs(features[0] - features[2])
            face_height = abs(features[1] - features[3])
            
            if face_width > 0 and face_height > 0:
                normalized_features = []
                for i in range(0, len(features), 2):
                    normalized_features.append(features[i] / face_width)
                    normalized_features.append(features[i+1] / face_height)
                return np.array(normalized_features)
        
        return np.array(features)

    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two face feature vectors"""
        if len(features1) != len(features2):
            return 0.0
        
        # Use cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)  # Ensure non-negative

    def register_face(self, name):
        """Register a new face to the database"""
        print(f"Registering face for: {name}")
        print("Look at the camera and press SPACE when ready, ESC to cancel")
        
        face_samples = []
        required_samples = 5
        
        while len(face_samples) < required_samples:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # Draw instructions
            cv2.putText(frame, f"Samples: {len(face_samples)}/{required_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture, ESC: Cancel", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw face mesh
                    self.mp_drawing.draw_landmarks(
                        frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # Space to capture
                        features = self.extract_face_features(frame, face_landmarks)
                        if len(features) > 0:
                            face_samples.append(features)
                            print(f"Sample {len(face_samples)} captured")
                    elif key == 27:  # ESC to cancel
                        print("Registration cancelled")
                        return False
            
            cv2.imshow('Register Face', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("Registration cancelled")
                return False
        
        # Average the samples
        avg_features = np.mean(face_samples, axis=0)
        self.known_faces[name] = avg_features
        self.save_face_database()
        
        print(f"Face registered successfully for {name}")
        return True

    def recognize_face(self, frame):
        """Recognize faces in the frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        recognized_names = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract features
                features = self.extract_face_features(frame, face_landmarks)
                
                if len(features) > 0:
                    # Compare with known faces
                    best_match = None
                    best_similarity = 0
                    
                    for name, known_features in self.known_faces.items():
                        similarity = self.calculate_similarity(features, known_features)
                        if similarity > best_similarity and similarity > self.recognition_threshold:
                            best_similarity = similarity
                            best_match = name
                    
                    if best_match:
                        recognized_names.append((best_match, best_similarity))
                        # Draw recognition result
                        cv2.putText(frame, f"{best_match} ({best_similarity:.2f})", 
                                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Unknown Person", 
                                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)
        
        return recognized_names

    def unlock_door(self, person_name):
        """Unlock the door for authorized person"""
        current_time = time.time()
        
        # Prevent rapid unlocking
        if current_time - self.last_unlock_time < 2:
            return
        
        print(f"ACCESS GRANTED: {person_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Activate relay (unlock door)
        GPIO.output(self.LOCK_PIN, GPIO.HIGH)
        self.last_unlock_time = current_time
        
        # Schedule lock after duration
        def lock_door():
            time.sleep(self.unlock_duration)
            GPIO.output(self.LOCK_PIN, GPIO.LOW)
            print("Door locked automatically")
        
        import threading
        threading.Thread(target=lock_door, daemon=True).start()

    def save_face_database(self):
        """Save face database to file"""
        try:
            with open(self.face_db_file, 'wb') as f:
                pickle.dump(self.known_faces, f)
        except Exception as e:
            print(f"Error saving database: {e}")

    def load_face_database(self):
        """Load face database from file"""
        try:
            if os.path.exists(self.face_db_file):
                with open(self.face_db_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                print(f"Loaded {len(self.known_faces)} faces from database")
            else:
                print("No existing face database found")
        except Exception as e:
            print(f"Error loading database: {e}")
            self.known_faces = {}

    def show_database(self):
        """Show registered faces"""
        print(f"\nRegistered faces ({len(self.known_faces)}):")
        for i, name in enumerate(self.known_faces.keys(), 1):
            print(f"{i}. {name}")
        print()

    def run(self):
        """Main loop"""
        print("Face Recognition Door Lock Active")
        print("Looking for authorized faces...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Recognize faces
                recognized = self.recognize_face(frame)
                
                # Check for authorized access
                for name, confidence in recognized:
                    self.unlock_door(name)
                
                # Display status
                status = "LOCKED"
                status_color = (0, 0, 255)  # Red
                if time.time() - self.last_unlock_time < self.unlock_duration:
                    status = "UNLOCKED"
                    status_color = (0, 255, 0)  # Green
                
                cv2.putText(frame, f"Status: {status}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(frame, f"Registered: {len(self.known_faces)} faces", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show GPIO status
                gpio_text = "GPIO: Real" if GPIO_AVAILABLE else "GPIO: Mock"
                cv2.putText(frame, gpio_text, 
                           (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow('Face Recognition Door Lock', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    name = input("Enter name to register: ").strip()
                    if name:
                        self.register_face(name)
                elif key == ord('s'):
                    self.show_database()
                elif key == ord('d'):
                    name = input("Enter name to delete: ").strip()
                    if name in self.known_faces:
                        del self.known_faces[name]
                        self.save_face_database()
                        print(f"Deleted {name} from database")
                    else:
                        print(f"Name {name} not found in database")
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.output(self.LOCK_PIN, GPIO.LOW)  # Ensure door is locked
        GPIO.cleanup()
        print("System shutdown complete")

if __name__ == "__main__":
    # Create and run the door lock system
    door_lock = FaceDoorLock()
    door_lock.run()