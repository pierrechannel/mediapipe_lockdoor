import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from datetime import datetime
import time
from face_processing import FaceProcessor
from api_handler import APIHandler

class FaceRecognitionDoorLock:
    def __init__(self, api_base_url=None, api_headers=None):
        # Initialize components
        self.api_handler = APIHandler(api_base_url, api_headers)
        self.face_processor = FaceProcessor()
        
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
    
    def sync_api_users(self):
        """Synchronize users from API with local authorized faces"""
        print("Syncing users from API...")
        
        api_response = self.api_handler.get_system_mode_users()
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
            
            # Detect faces and get face encodings
            face_encoding = self.face_processor.process_frame_for_registration(frame, rgb_frame)
            
            if face_encoding is not None:
                # Show status
                cv2.putText(frame, f"Sample {sample_count + 1}/{required_samples}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to capture", 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Add Authorized Face', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and face_encoding is not None:
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
            similarity = self.face_processor.calculate_face_similarity(face_encoding, auth_encoding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity > self.recognition_threshold:
            return best_match, best_similarity
        
        return None, best_similarity
    
    def process_access_attempt(self, frame, person_name, similarity):
        """Handle access attempt logging with proper user ID conversion"""
        status = 1 if person_name else 0  # 1 for recognized, 0 for unrecognized
        
        # Determine user ID
        if person_name:
            user_id = None
            for uid, user_info in self.api_users.items():
                if user_info['name'] == person_name:
                    user_id = uid
                    break
            
            # Use numeric ID if available, otherwise use 0
            log_user_id = user_id if user_id else '0'
        else:
            log_user_id = '0'  # Unknown user
        
        # Log the access attempt
        return self.api_handler.log_access(log_user_id, frame, status)
    
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
            
            # Process frame for recognition
            recognition_result = self.face_processor.process_frame_for_recognition(
                frame, rgb_frame, self.authorized_faces, self.recognition_threshold,
                consecutive_recognitions, recognition_threshold_count
            )
            
            if recognition_result:
                person_name, similarity, bbox = recognition_result
                self.process_access_attempt(frame, person_name, similarity)
                
                if person_name:
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
    
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_processor.cleanup()