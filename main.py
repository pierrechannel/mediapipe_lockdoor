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
        self.auto_registration_mode = False
        self.pending_registrations = []  # Queue for auto-registration
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Load users from API and auto-register
        self.sync_users_from_api()
        
        gpio_status = "enabled" if GPIO_AVAILABLE else "mocked"
        print(f"API-Integrated Face Recognition Door Lock System Initialized (GPIO {gpio_status})")
        print("Commands: 'a' - Auto-register all users, 'q' - Quit, 's' - Show database, 'sync' - Sync with API")

    def sync_users_from_api(self):
        """Fetch users from the API and update local cache"""
        try:
            # Fetch users from API
            response = requests.get(f"{self.api_base_url}/administration/warehouse_users", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('statusCode') == 200 and 'result' in data:
                    users = data['result']['data']
                    
                    # Update user database and identify new users
                    new_users = []
                    for user in users:
                        user_id = user['WAREHOUSE_USER_ID']
                        user_info = {
                            'id': user_id,
                            'name': f"{user['PRENOM']} {user['NOM']}",
                            'email': user['EMAIL'],
                            'phone': user['TELEPHONE'],
                            'photo': user['PHOTO'],
                            'profile': user.get('profil', {}).get('DESCRIPTION_PROFIL', 'USER'),
                            'date_save': user['DATE_SAVE']
                        }
                        
                        # Check if user is new or doesn't have face data
                        if user_id not in self.user_database or user_id not in self.known_faces:
                            new_users.append(user_id)
                        
                        self.user_database[user_id] = user_info
                    
                    print(f"Synced {len(users)} users from API")
                    
                    # Auto-register faces for users with photos
                    if new_users:
                        print(f"Found {len(new_users)} users for auto-registration")
                        self.auto_register_users_from_photos(new_users)
                    
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

    def auto_register_users_from_photos(self, user_ids):
        """Automatically register faces from user photos in the API"""
        for user_id in user_ids:
            if user_id in self.user_database:
                user = self.user_database[user_id]
                photo_url = user.get('photo')
                
                if photo_url and photo_url.strip():
                    print(f"Auto-registering face for {user['name']} from photo...")
                    success = self.register_face_from_photo(user_id, photo_url)
                    if success:
                        print(f"✓ Successfully registered {user['name']}")
                    else:
                        print(f"✗ Failed to register {user['name']} - will need manual registration")
                        self.pending_registrations.append(user_id)
                else:
                    print(f"No photo available for {user['name']} - adding to manual registration queue")
                    self.pending_registrations.append(user_id)

    def register_face_from_photo(self, user_id, photo_url):
        """Register face from user's photo URL"""
        try:
            # Download photo from URL
            if photo_url.startswith('data:image'):
                # Handle base64 encoded images
                header, data = photo_url.split(',', 1)
                image_data = base64.b64decode(data)
                image = Image.open(BytesIO(image_data))
            else:
                # Handle URL images
                response = requests.get(photo_url, timeout=10)
                if response.status_code != 200:
                    print(f"Failed to download photo: HTTP {response.status_code}")
                    return False
                image = Image.open(BytesIO(response.content))
            
            # Convert PIL image to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process image for face detection
            rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                # Use the first detected face
                face_landmarks = results.multi_face_landmarks[0]
                features = self.extract_face_features(image_cv, face_landmarks)
                
                if len(features) > 0:
                    # Store the face features
                    self.known_faces[user_id] = features
                    self.save_face_database()
                    return True
                else:
                    print(f"Could not extract features from photo for user {user_id}")
            else:
                print(f"No face detected in photo for user {user_id}")
                
        except Exception as e:
            print(f"Error processing photo for user {user_id}: {e}")
        
        return False

    def start_manual_registration_mode(self):
        """Start interactive registration for users without photos"""
        if not self.pending_registrations:
            print("No users pending manual registration")
            return
        
        print(f"Starting manual registration for {len(self.pending_registrations)} users")
        print("Press 'n' for next user, 'skip' to skip current user, 'q' to quit registration")
        
        self.auto_registration_mode = True
        self.current_registration_index = 0

    def handle_manual_registration(self, frame):
        """Handle manual registration during main loop"""
        if not self.auto_registration_mode or not self.pending_registrations:
            return frame
        
        if self.current_registration_index >= len(self.pending_registrations):
            print("Manual registration complete!")
            self.auto_registration_mode = False
            self.pending_registrations = []
            return frame
        
        user_id = self.pending_registrations[self.current_registration_index]
        user = self.user_database[user_id]
        
        # Draw registration interface
        cv2.rectangle(frame, (10, 10), (630, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (630, 150), (0, 255, 255), 2)
        
        cv2.putText(frame, "MANUAL REGISTRATION MODE", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"User: {user['name']} ({user_id})", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Role: {user['profile']}", 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "SPACE: Register face | N: Next | S: Skip | Q: Quit", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f"Progress: {self.current_registration_index + 1}/{len(self.pending_registrations)}", 
                   (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

    def register_current_user_face(self, frame):
        """Register face for current user in manual mode"""
        if not self.auto_registration_mode or not self.pending_registrations:
            return False
        
        user_id = self.pending_registrations[self.current_registration_index]
        user = self.user_database[user_id]
        
        # Process frame for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Use the first detected face
            face_landmarks = results.multi_face_landmarks[0]
            features = self.extract_face_features(frame, face_landmarks)
            
            if len(features) > 0:
                self.known_faces[user_id] = features
                self.save_face_database()
                print(f"✓ Face registered for {user['name']}")
                
                # Move to next user
                self.current_registration_index += 1
                return True
            else:
                print("Could not extract face features. Please try again.")
        else:
            print("No face detected. Please look at the camera.")
        
        return False

    def log_access_to_api(self, user_id, image_frame, status=1):
        """Log access attempt to the API"""
        try:
            # Convert frame to JPEG image bytes
            _, buffer = cv2.imencode('.jpg', image_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Prepare multipart form data
            files = {
                'IMAGE': ('access_image.jpg', buffer.tobytes(), 'image/jpeg')
            }
            
            # Prepare form data (non-image fields)
            data = {
                'WAREHOUSE_USER_ID': str(user_id),
                'STATUT': str(status),  # 1 for success, 0 for failure
                'DATE_SAVE': datetime.now().isoformat()
            }
            
            # Prepare headers for multipart upload (remove Content-Type to let requests set it)
            upload_headers = self.headers.copy()
            if 'Content-Type' in upload_headers:
                del upload_headers['Content-Type']
            
            # Send to API with multipart form data
            response = requests.post(
                f"{self.api_base_url}/warehouse_acces/create", 
                files=files,
                data=data,
                headers=upload_headers
            )
            
            if response.status_code == 200:
                print("Access logged to API successfully")
                return True
            else:
                print(f"Failed to log access: HTTP {response.status_code}")
                if response.text:
                    print(f"Response: {response.text}")
                
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
                                   (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Role: {user['profile']}", 
                                   (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Unknown Person", 
                                   (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
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
        if self.pending_registrations:
            print(f"Pending Manual Registration: {len(self.pending_registrations)}")
        print()

    def run(self):
        """Main loop"""
        print("API-Integrated Face Recognition Door Lock Active")
        print("Auto-registration completed. Looking for authorized faces...")
        
        if self.pending_registrations:
            print(f"Note: {len(self.pending_registrations)} users need manual registration. Press 'a' to start.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Handle manual registration overlay
                if self.auto_registration_mode:
                    frame = self.handle_manual_registration(frame)
                else:
                    # Normal recognition mode
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
                
                # Status information
                if not self.auto_registration_mode:
                    cv2.putText(frame, f"Status: {status}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                    cv2.putText(frame, f"Users: {len(self.user_database)} | Faces: {len(self.known_faces)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if self.pending_registrations:
                        cv2.putText(frame, f"Pending Registration: {len(self.pending_registrations)}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
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
                elif key == ord('a') and not self.auto_registration_mode:
                    self.start_manual_registration_mode()
                elif key == ord('s'):
                    self.show_database()
                elif key == ord('1'):  # Sync with API
                    print("Syncing with API...")
                    self.sync_users_from_api()
                elif self.auto_registration_mode:
                    if key == ord(' '):  # Space to register face
                        self.register_current_user_face(frame)
                    elif key == ord('n'):  # Next user
                        self.current_registration_index += 1
                    elif key == ord('S'):  # Skip user (capital S)
                        print(f"Skipped {self.user_database[self.pending_registrations[self.current_registration_index]]['name']}")
                        self.current_registration_index += 1
                    elif key == ord('q'):  # Quit registration
                        self.auto_registration_mode = False
        
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