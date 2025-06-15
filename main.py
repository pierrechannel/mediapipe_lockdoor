import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
import requests
import json
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

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

class APIIntegratedFaceDoorLock:
    def __init__(self, api_base_url, api_key=None):
        # API Configuration
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Face detection and mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # GPIO setup for door lock
        self.LOCK_PIN = 18
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.LOCK_PIN, GPIO.OUT)
        GPIO.output(self.LOCK_PIN, GPIO.LOW)
        
        # Face database (local cache + API sync)
        self.known_faces = {}
        self.user_database = {}  # Store user info from API
        self.face_db_file = "face_database.pkl"
        self.load_face_database()
        
        # Settings
        self.recognition_threshold = 0.6
        self.unlock_duration = 5
        self.last_unlock_time = 0
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Load users from API
        self.sync_users_from_api()
        
        gpio_status = "enabled" if GPIO_AVAILABLE else "mocked"
        print(f"API-Integrated Face Recognition Door Lock System Initialized (GPIO {gpio_status})")
        print("Commands: 'r' - Register face, 'q' - Quit, 's' - Show database, 'sync' - Sync with API")

    def sync_users_from_api(self):
        """Fetch users from the API and update local cache"""
        try:
            # Fetch users from API
            response = requests.get(f"{self.api_base_url}/administration/warehouse_users", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('statusCode') == 200 and 'result' in data:
                    users = data['result']['data']
                    
                    # Update user database
                    for user in users:
                        user_id = user['WAREHOUSE_USER_ID']
                        self.user_database[user_id] = {
                            'id': user_id,
                            'name': f"{user['PRENOM']} {user['NOM']}",
                            'email': user['EMAIL'],
                            'phone': user['TELEPHONE'],
                            'photo': user['PHOTO'],
                            'profile': user.get('profil', {}).get('DESCRIPTION_PROFIL', 'USER'),
                            'date_save': user['DATE_SAVE']
                        }
                    
                    print(f"Synced {len(users)} users from API")
                    return True
                else:
                    print(f"API Error: {data.get('message', 'Unknown error')}")
            else:
                print(f"Failed to fetch users: HTTP {response.status_code}")
                
        except requests.RequestException as e:
            print(f"Network error while syncing users: {e}")
        except Exception as e:
            print(f"Error syncing users: {e}")
        
        return False

    def log_access_to_api(self, user_id, image_frame, status=1):
        """Log access attempt to the API"""
        try:
            # Convert frame to base64 for API upload
            _, buffer = cv2.imencode('.jpg', image_frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare access log data
            access_data = {
                'WAREHOUSE_USER_ID': user_id,
                'IMAGE_DATA': image_base64,  # Base64 encoded image
                'STATUT': status,  # 1 for success, 0 for failure
                'DATE_SAVE': datetime.now().isoformat()
            }
            
            # Send to API
            response = requests.post(
                f"{self.api_base_url}/warehouse_acces/create", 
                json=access_data, 
                headers=self.headers
            )
            
            if response.status_code == 200:
                print("Access logged to API successfully")
                return True
            else:
                print(f"Failed to log access: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Error logging access to API: {e}")
        
        return False

    def extract_face_features(self, image, face_landmarks):
        """Extract facial features from MediaPipe landmarks"""
        h, w = image.shape[:2]
        
        # Key facial landmarks indices
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
        
        # Normalize features
        if len(features) >= 4:
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
        return max(0.0, similarity)

    def register_face_for_user(self, user_id):
        """Register a face for an existing user from the API"""
        if user_id not in self.user_database:
            print(f"User ID {user_id} not found in database. Please sync with API first.")
            return False
        
        user = self.user_database[user_id]
        print(f"Registering face for: {user['name']} (ID: {user_id})")
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
            cv2.putText(frame, f"User: {user['name']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Samples: {len(face_samples)}/{required_samples}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture, ESC: Cancel", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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
        
        # Average the samples and store with user ID
        avg_features = np.mean(face_samples, axis=0)
        self.known_faces[user_id] = avg_features
        self.save_face_database()
        
        print(f"Face registered successfully for {user['name']} (ID: {user_id})")
        return True

    def recognize_face(self, frame):
        """Recognize faces in the frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        recognized_users = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract features
                features = self.extract_face_features(frame, face_landmarks)
                
                if len(features) > 0:
                    # Compare with known faces
                    best_match_id = None
                    best_similarity = 0
                    
                    for user_id, known_features in self.known_faces.items():
                        similarity = self.calculate_similarity(features, known_features)
                        if similarity > best_similarity and similarity > self.recognition_threshold:
                            best_similarity = similarity
                            best_match_id = user_id
                    
                    if best_match_id and best_match_id in self.user_database:
                        user = self.user_database[best_match_id]
                        recognized_users.append((best_match_id, user, best_similarity))
                        
                        # Draw recognition result
                        cv2.putText(frame, f"{user['name']} ({best_similarity:.2f})", 
                                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Role: {user['profile']}", 
                                   (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Unknown Person", 
                                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)
        
        return recognized_users

    def unlock_door(self, user_id, user_info, frame):
        """Unlock the door for authorized person"""
        current_time = time.time()
        
        # Prevent rapid unlocking
        if current_time - self.last_unlock_time < 2:
            return
        
        print(f"ACCESS GRANTED: {user_info['name']} (ID: {user_id}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Log access to API
        self.log_access_to_api(user_id, frame, status=1)
        
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
                print(f"Loaded {len(self.known_faces)} faces from local database")
            else:
                print("No existing local face database found")
        except Exception as e:
            print(f"Error loading database: {e}")
            self.known_faces = {}

    def show_database(self):
        """Show registered users and faces"""
        print(f"\nAPI Users ({len(self.user_database)}):")
        for user_id, user in self.user_database.items():
            face_status = "✓ Registered" if user_id in self.known_faces else "✗ No face data"
            print(f"ID: {user_id} | {user['name']} | {user['profile']} | {face_status}")
        
        print(f"\nRegistered Faces: {len(self.known_faces)}")
        print()

    def run(self):
        """Main loop"""
        print("API-Integrated Face Recognition Door Lock Active")
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
                for user_id, user_info, confidence in recognized:
                    self.unlock_door(user_id, user_info, frame)
                
                # Display status
                status = "LOCKED"
                status_color = (0, 0, 255)  # Red
                if time.time() - self.last_unlock_time < self.unlock_duration:
                    status = "UNLOCKED"
                    status_color = (0, 255, 0)  # Green
                
                cv2.putText(frame, f"Status: {status}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(frame, f"Users: {len(self.user_database)} | Faces: {len(self.known_faces)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show GPIO and API status
                gpio_text = "GPIO: Real" if GPIO_AVAILABLE else "GPIO: Mock"
                cv2.putText(frame, gpio_text, 
                           (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, "API: Connected", 
                           (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow('API-Integrated Face Recognition Door Lock', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    try:
                        user_id = int(input("Enter User ID to register face: "))
                        self.register_face_for_user(user_id)
                    except ValueError:
                        print("Please enter a valid numeric User ID")
                elif key == ord('s'):
                    self.show_database()
                elif key == ord('1'):  # Sync with API
                    print("Syncing with API...")
                    self.sync_users_from_api()
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.output(self.LOCK_PIN, GPIO.LOW)
        GPIO.cleanup()
        print("System shutdown complete")

if __name__ == "__main__":
    # Configuration
    API_BASE_URL = "https://apps.mediabox.bi:26875/"  # Replace with your actual API URL
    API_KEY = "your_api_key_here"  # Replace with your actual API key if required
    
    # Create and run the API-integrated door lock system
    door_lock = APIIntegratedFaceDoorLock(API_BASE_URL, API_KEY)
    door_lock.run()