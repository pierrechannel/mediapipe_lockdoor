import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from datetime import datetime
import time
import requests
import json

class FaceRecognitionDoorLock:
    def __init__(self, api_base_url=None, api_headers=None):
        # Initialize MediaPipe Face Detection and Face Mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Initialize face detection and mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # API configuration
        self.api_base_url = api_base_url
        self.headers = api_headers or {'Content-Type': 'application/json'}
        
        # Storage for authorized faces
        self.authorized_faces = {}
        self.api_users = {}  # Store users from API
        self.face_encodings_file = "authorized_faces.pkl"
        self.load_authorized_faces()
        
        # Recognition parameters
        self.recognition_threshold = 0.6
        self.unlock_duration = 3  # seconds
        self.last_unlock_time = 0
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
    
    def get_system_mode_users(self):
        """Get registration requests with proper response handling
        
        Returns:
            dict: The API response data if successful, None otherwise
            (including when no data is found)
        """
        if not self.api_base_url:
            print("API base URL not configured")
            return None
            
        try:
            response = requests.get(
                f"{self.api_base_url}/administration/warehouse_users/checkmode/allregistration",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
                      
            # Validate response structure is a dictionary
            if not isinstance(data, dict):
                raise ValueError("Invalid API response format - expected a dictionary")
                         
            # Handle case when no data is found
            if data.get("message") == "Aucune donnée trouvée" and data.get("result") is None:
                print("No registration data found")
                return None
                         
            # Check status code in response
            if data.get('statusCode') != 200:
                error_msg = data.get('message', 'No error message provided')
                print(f"API Error: {error_msg}")
                return None
                         
            # Additional validation - check if expected data fields are present
            if 'result' not in data:
                print("API response missing expected 'result' field")
                return None
                         
            return data
                     
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return None
        except ValueError as e:
            print(f"Invalid JSON response: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None
    
    def sync_api_users(self):
        """Synchronize users from API with local authorized faces"""
        print("Syncing users from API...")
        
        api_response = self.get_system_mode_users()
        if not api_response:
            print("Failed to retrieve users from API")
            return False
        
        user_data = api_response.get('result')
        if not user_data:
            print("No user data in API response")
            return False
        
        # Handle single user response (the API returns one user with MODE=2)
        if isinstance(user_data, dict):
            user_info = user_data.get('user', {})
            
            # Extract correct user ID and name from API response
            user_id = user_info.get('WAREHOUSE_USER_ID')
            nom = user_info.get('NOM', '')
            prenom = user_info.get('PRENOM', '')
            
            # Construct full name
            if nom and prenom:
                user_name = f"{prenom} {nom}"
            elif nom:
                user_name = nom
            elif prenom:
                user_name = prenom
            else:
                user_name = f"User_{user_id}"
            
            # Store API user info
            self.api_users[user_id] = {
                'name': user_name,
                'registration_data': user_data,
                'user_info': user_info,
                'email': user_info.get('EMAIL', ''),
                'telephone': user_info.get('TELEPHONE', ''),
                'photo_url': user_info.get('PHOTO', ''),
                'face_registration_date': user_info.get('FACE_REGISTRATION_DATE'),
                'face_encoding_path': user_info.get('FACE_ENCODING_PATH')
            }
            
            print(f"Found API user: {user_name} (ID: {user_id})")
            print(f"  Email: {user_info.get('EMAIL', 'N/A')}")
            print(f"  Phone: {user_info.get('TELEPHONE', 'N/A')}")
            print(f"  Registration Status: {user_info.get('REGISTRATION_STATUS', 'N/A')}")
            
            # Check if this user already has face encoding
            if user_name not in self.authorized_faces:
                print(f"User {user_name} needs face registration")
                return True
            else:
                print(f"User {user_name} already has face encoding")
                return True
        
        return False
    
    def register_api_user_face(self, user_id):
        """Register face for a specific API user"""
        if user_id not in self.api_users:
            print(f"User ID {user_id} not found in API users")
            return False
        
        user_info = self.api_users[user_id]
        user_name = user_info['name']
        
        print(f"Registering face for API user: {user_name}")
        return self.add_authorized_face(user_name)
    
    def extract_face_landmarks(self, image, face_landmarks):
        """Extract key facial landmarks as feature vector"""
        h, w = image.shape[:2]
        landmarks = []
        
        # Extract key facial points (eyes, nose, mouth corners, etc.)
        key_points = [
            # Left eye
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            # Right eye  
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            # Nose
            1, 2, 5, 4, 6, 168, 8, 9, 10, 151, 195, 197, 196, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 240, 309, 415, 310, 311, 312, 13, 82, 81, 80, 78,
            # Mouth
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318
        ]
        
        for point_idx in key_points:
            if point_idx < len(face_landmarks.landmark):
                x = int(face_landmarks.landmark[point_idx].x * w)
                y = int(face_landmarks.landmark[point_idx].y * h)
                landmarks.extend([x, y])
        
        return np.array(landmarks)
    
    def calculate_face_similarity(self, face1, face2):
        """Calculate similarity between two face encodings"""
        if len(face1) != len(face2):
            return 0
        
        # Normalize the vectors
        face1_norm = face1 / (np.linalg.norm(face1) + 1e-6)
        face2_norm = face2 / (np.linalg.norm(face2) + 1e-6)
        
        # Calculate cosine similarity
        similarity = np.dot(face1_norm, face2_norm)
        return max(0, similarity)  # Ensure non-negative
    
    def add_authorized_face(self, name):
        """Add a new authorized face to the system"""
        print(f"Adding authorized face for: {name}")
        print("Look at the camera and press SPACE when ready, or 'q' to cancel")
        
        face_samples = []
        sample_count = 0
        required_samples = 5
        
        while sample_count < required_samples:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    # Draw bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, c = frame.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                           int(bboxC.width * w), int(bboxC.height * h)
                    
                    cv2.rectangle(frame, bbox, (0, 255, 0), 2)
                    
                    # Get face mesh for this detection
                    mesh_results = self.face_mesh.process(rgb_frame)
                    if mesh_results.multi_face_landmarks:
                        face_landmarks = mesh_results.multi_face_landmarks[0]
                        face_encoding = self.extract_face_landmarks(rgb_frame, face_landmarks)
                        
                        # Show status
                        cv2.putText(frame, f"Sample {sample_count + 1}/{required_samples}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, "Press SPACE to capture", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Add Authorized Face', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and results.detections:
                mesh_results = self.face_mesh.process(rgb_frame)
                if mesh_results.multi_face_landmarks:
                    face_landmarks = mesh_results.multi_face_landmarks[0]
                    face_encoding = self.extract_face_landmarks(rgb_frame, face_landmarks)
                    face_samples.append(face_encoding)
                    sample_count += 1
                    print(f"Captured sample {sample_count}/{required_samples}")
            elif key == ord('q'):
                print("Cancelled adding face")
                cv2.destroyWindow('Add Authorized Face')
                return False
        
        cv2.destroyWindow('Add Authorized Face')
        
        # Average the face samples
        if face_samples:
            avg_encoding = np.mean(face_samples, axis=0)
            self.authorized_faces[name] = avg_encoding
            self.save_authorized_faces()
            print(f"Successfully added {name} to authorized faces")
            return True
        
        return False
    
    def save_authorized_faces(self):
        """Save authorized faces to file"""
        with open(self.face_encodings_file, 'wb') as f:
            pickle.dump(self.authorized_faces, f)
    
    def load_authorized_faces(self):
        """Load authorized faces from file"""
        if os.path.exists(self.face_encodings_file):
            try:
                with open(self.face_encodings_file, 'rb') as f:
                    self.authorized_faces = pickle.load(f)
                print(f"Loaded {len(self.authorized_faces)} authorized faces")
            except:
                print("Could not load authorized faces file")
                self.authorized_faces = {}
        else:
            print("No authorized faces file found")
    
    def recognize_face(self, face_encoding):
        """Recognize a face against authorized faces"""
        if not self.authorized_faces:
            return None, 0
        
        best_match = None
        best_similarity = 0
        
        for name, auth_encoding in self.authorized_faces.items():
            similarity = self.calculate_face_similarity(face_encoding, auth_encoding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity > self.recognition_threshold:
            return best_match, best_similarity
        
        return None, best_similarity
    
    def unlock_door(self, person_name):
        """Simulate door unlocking"""
        current_time = time.time()
        if current_time - self.last_unlock_time > self.unlock_duration:
            self.last_unlock_time = current_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"🔓 DOOR UNLOCKED for {person_name} at {timestamp}")
            # Here you would interface with actual door lock hardware
            return True
        return False
    
    def run_door_lock_system(self):
        """Main door lock system loop"""
        print("Face Recognition Door Lock System Started")
        print("Press 'q' to quit")
        
        consecutive_recognitions = {}
        recognition_threshold_count = 3  # Need 3 consecutive recognitions
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_detection.process(rgb_frame)
            
            # Reset recognition counters
            current_recognitions = set()
            
            if results.detections:
                for detection in results.detections:
                    # Draw bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, c = frame.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                           int(bboxC.width * w), int(bboxC.height * h)
                    
                    # Get face mesh for recognition
                    mesh_results = self.face_mesh.process(rgb_frame)
                    if mesh_results.multi_face_landmarks:
                        face_landmarks = mesh_results.multi_face_landmarks[0]
                        face_encoding = self.extract_face_landmarks(rgb_frame, face_landmarks)
                        
                        # Recognize face
                        person_name, similarity = self.recognize_face(face_encoding)
                        
                        if person_name:
                            current_recognitions.add(person_name)
                            
                            # Update consecutive recognition counter
                            consecutive_recognitions[person_name] = consecutive_recognitions.get(person_name, 0) + 1
                            
                            # Draw green box for authorized person
                            cv2.rectangle(frame, bbox, (0, 255, 0), 3)
                            
                            # Check if should unlock
                            if consecutive_recognitions[person_name] >= recognition_threshold_count:
                                if self.unlock_door(person_name):
                                    cv2.putText(frame, f"UNLOCKED: {person_name}", 
                                              (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.8, (0, 255, 0), 2)
                                else:
                                    cv2.putText(frame, f"DOOR OPEN: {person_name}", 
                                              (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.8, (0, 255, 255), 2)
                            else:
                                cv2.putText(frame, f"Recognizing: {person_name} ({consecutive_recognitions[person_name]}/{recognition_threshold_count})", 
                                          (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.6, (0, 255, 0), 2)
                            
                            cv2.putText(frame, f"Confidence: {similarity:.2f}", 
                                      (bbox[0], bbox[1] + bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 255, 0), 1)
                        else:
                            # Draw red box for unauthorized person
                            cv2.rectangle(frame, bbox, (0, 0, 255), 3)
                            cv2.putText(frame, "UNAUTHORIZED", 
                                      (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.8, (0, 0, 255), 2)
                            cv2.putText(frame, f"Similarity: {similarity:.2f}", 
                                      (bbox[0], bbox[1] + bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 0, 255), 1)
            
            # Reset counters for people not currently detected
            for person in list(consecutive_recognitions.keys()):
                if person not in current_recognitions:
                    consecutive_recognitions[person] = 0
            
            # Display system info
            cv2.putText(frame, f"Authorized Users: {len(self.authorized_faces)}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"API Users: {len(self.api_users)}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Face Recognition Door Lock", 
                      (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Door Lock System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    def show_menu(self):
        """Show system menu"""
        while True:
            print("\n" + "="*60)
            print("FACE RECOGNITION DOOR LOCK SYSTEM WITH API INTEGRATION")
            print("="*60)
            print("1. Sync Users from API")
            print("2. Add Authorized Face (Manual)")
            print("3. Register Face for API User")
            print("4. List Authorized Users")
            print("5. List API Users")
            print("6. Remove Authorized User")
            print("7. Start Door Lock System")
            print("8. Exit")
            print("="*60)
            
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == '1':
                self.sync_api_users()
            
            elif choice == '2':
                name = input("Enter name for new authorized user: ").strip()
                if name:
                    self.add_authorized_face(name)
                else:
                    print("Invalid name")
            
            elif choice == '3':
                if self.api_users:
                    print("\nAPI Users needing face registration:")
                    available_users = []
                    for user_id, user_info in self.api_users.items():
                        user_name = user_info['name']
                        if user_name not in self.authorized_faces:
                            available_users.append((user_id, user_name))
                            print(f"{len(available_users)}. {user_name} (ID: {user_id})")
                    
                    if available_users:
                        try:
                            idx = int(input("Enter number to register face for: ")) - 1
                            if 0 <= idx < len(available_users):
                                user_id, user_name = available_users[idx]
                                self.register_api_user_face(user_id)
                            else:
                                print("Invalid selection")
                        except ValueError:
                            print("Invalid input")
                    else:
                        print("All API users already have registered faces")
                else:
                    print("No API users found. Please sync users first.")
            
            elif choice == '4':
                if self.authorized_faces:
                    print("\nAuthorized Users:")
                    for i, name in enumerate(self.authorized_faces.keys(), 1):
                        print(f"{i}. {name}")
                else:
                    print("No authorized users found")
            
            elif choice == '5':
                if self.api_users:
                    print("\nAPI Users:")
                    for user_id, user_info in self.api_users.items():
                        user_name = user_info['name']
                        email = user_info.get('email', 'N/A')
                        phone = user_info.get('telephone', 'N/A')
                        has_face = "✓" if user_name in self.authorized_faces else "✗"
                        print(f"ID: {user_id}")
                        print(f"  Name: {user_name}")
                        print(f"  Email: {email}")
                        print(f"  Phone: {phone}")
                        print(f"  Face Registered: {has_face}")
                        print(f"  Registration Date: {user_info.get('face_registration_date', 'N/A')}")
                        print("-" * 50)
                else:
                    print("No API users found. Please sync users first.")
            
            elif choice == '6':
                if self.authorized_faces:
                    print("\nAuthorized Users:")
                    users = list(self.authorized_faces.keys())
                    for i, name in enumerate(users, 1):
                        print(f"{i}. {name}")
                    
                    try:
                        idx = int(input("Enter number to remove: ")) - 1
                        if 0 <= idx < len(users):
                            removed_user = users[idx]
                            del self.authorized_faces[removed_user]
                            self.save_authorized_faces()
                            print(f"Removed {removed_user} from authorized users")
                        else:
                            print("Invalid selection")
                    except ValueError:
                        print("Invalid input")
                else:
                    print("No authorized users to remove")
            
            elif choice == '7':
                if self.authorized_faces:
                    self.run_door_lock_system()
                else:
                    print("No authorized faces found. Please add authorized users first.")
            
            elif choice == '8':
                break
            
            else:
                print("Invalid choice")
    
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    # Check if required packages are available
    try:
        import mediapipe
        import cv2
        import numpy
        import pickle
        import requests
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Install required packages with:")
        print("pip install mediapipe opencv-python numpy requests")
        return
    
    # Configure API settings
    API_BASE_URL = 'https://apps.mediabox.bi:26875'
    
    api_headers = None
    if API_BASE_URL:
        # You can configure headers here if needed (e.g., authentication tokens)
        api_headers = {
            'Content-Type': 'application/json',
            # 'Authorization': 'Bearer your-token-here'  # Add if needed
        }
    
    door_lock = FaceRecognitionDoorLock(api_base_url=API_BASE_URL, api_headers=api_headers)
    
    try:
        door_lock.show_menu()
    except KeyboardInterrupt:
        print("\nShutting down system...")
    finally:
        door_lock.cleanup()

if __name__ == "__main__":
    main()