import cv2
import numpy as np
import pickle
import os
from datetime import datetime
import time
import threading
import subprocess
from face_processing import FaceProcessor
from api_handler import APIHandler
from streaming_manager import StreamingManager
from flite_tts import FliteTTS

class FaceRecognitionDoorLock:
    def __init__(self, api_base_url=None, api_headers=None, headless=False, enable_tts=True, enable_streaming=True):
        # Initialize components
        self.api_handler = APIHandler(api_base_url, api_headers)
        self.face_processor = FaceProcessor()
        
        # System configuration
        self.headless = headless
        self.enable_tts = enable_tts
        
        # Initialize Flite Text-to-Speech
        if self.enable_tts:
            try:
                self.tts = FliteTTS(
                    voice='slt',  # Default voice
                    speech_rate=170,  # Normal speech rate
                    volume=0.8,  # 80% volume
                    timeout=10  # 10 second timeout
                )
                
                if not self.tts.flite_available:
                    print("Flite TTS not available - disabling")
                    self.enable_tts = False
                else:
                    print("Flite TTS initialized successfully")
                    # Test the voice
                    self.tts.test_voice()
                    
            except Exception as e:
                print(f"Failed to initialize Flite TTS: {e}")
                self.enable_tts = False
        
        # Storage for authorized faces
        self.authorized_faces = {}
        self.api_users = {}  # Store users from API
        self.face_encodings_file = "authorized_faces.pkl"
        self.api_users_file = "api_users.pkl"
        
        # Load both authorized faces and API users
        self.load_authorized_faces()
        self.load_api_users()
        
        # Recognition parameters
        self.recognition_threshold = 0.6
        self.unlock_duration = 3  # seconds
        self.last_unlock_time = 0
        self.mode_check_interval = 10  # Check API every 10 seconds
        self.last_mode_check = 0
        
        # ACCESS LOGGING CONTROL - NEW
        self.last_access_log_time = 0
        self.access_log_cooldown = 5  # Minimum seconds between access logs
        self.door_locked = True  # Track door state
        self.access_logged_for_current_session = False  # Track if we've logged for current recognition session
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        self.enable_streaming = enable_streaming
        self.streaming_manager = StreamingManager(
            api_base_url=api_base_url,
            api_headers=api_headers,
            enable_streaming=enable_streaming
        )

    def speak(self, text, priority=False):
        """Add text to speech queue using FliteTTS"""
        if not self.enable_tts:
            return
            
        if priority:
            self.tts.speak(text, priority=True)
        else:
            self.tts.speak(text)

    def speak_immediate(self, text):
        """Speak text immediately (blocking)"""
        if not self.enable_tts:
            return
            
        self.tts.stop_current_speech()  # Stop any ongoing speech
        self.tts.speak(text, priority=True)  # Will be processed immediately

    def start_system(self):
        """Start all system components including streaming"""
        # Start streaming service
        self.streaming_manager.start_streaming()
        print("Face recognition system with streaming started")
        if self.enable_tts:
            self.speak("System started with streaming enabled", priority=True)

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
            
            # Validate user_id
            if not user_id or user_id == 0:
                print(f"Invalid user ID received: {user_id}")
                return False
            
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
                'face_encoding_path': user_info.get('FACE_ENCODING_PATH'),
                'status': 'pending'  # Track registration status
            }
            
            # Save API users data to persist between restarts
            self.save_api_users()
            
            print(f"Found API user: {user_name} (ID: {user_id})")
            print(f"  Email: {user_info.get('EMAIL', 'N/A')}")
            print(f"  Phone: {user_info.get('TELEPHONE', 'N/A')}")
            print(f"  Registration Status: {user_info.get('REGISTRATION_STATUS', 'N/A')}")
            
            # Check if this user already has face encoding
            if user_name not in self.authorized_faces:
                print(f"User {user_name} needs face registration")
                self.speak(f"New user {user_name} detected. Face registration required.")
                return True
            else:
                print(f"User {user_name} already has face encoding")
                return True
        
        return False

    def run_door_lock_system(self):
        """Main door lock system loop"""
        self.start_system()  # Make sure this starts the streaming
        print("Face Recognition Door Lock System Started")
        if self.headless:
            print("Running in headless mode - no GUI display")
        else:
            print("Press 'q' to quit")
        
        self.speak("Face recognition door lock system started.", priority=True)
        
        consecutive_recognitions = {}
        recognition_threshold_count = 3  # Need 3 consecutive recognitions
        
        # For headless mode, we need a way to stop the loop
        if self.headless:
            print("Press Ctrl+C to stop the system")
        
        try:
            while True:
                # Check for API users periodically
                current_time = time.time()
                if current_time - self.last_mode_check > self.mode_check_interval:
                    if self.sync_api_users():
                        # New API users detected, attempt registration
                        for user_id, user_info in self.api_users.items():
                            if user_info.get('status') == 'pending':
                                self.register_api_user(user_id)
                    self.last_mode_check = current_time

                # Check if door should be locked again
                self.check_door_lock_status(current_time)

                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Only process recognition if door is locked
                if self.door_locked:
                    # Process frame for recognition
                    recognition_result = self.face_processor.process_frame_for_recognition(
                        frame, rgb_frame, self.authorized_faces, self.recognition_threshold,
                        consecutive_recognitions, recognition_threshold_count
                    )
                    
                    if recognition_result:
                        person_name, similarity, bbox = recognition_result
                        
                        # Only process access attempt if we haven't logged for this session
                        if not self.access_logged_for_current_session:
                            self.process_access_attempt(frame, person_name, similarity)
                            self.access_logged_for_current_session = True
                        
                        if person_name:
                            # Check if should unlock
                            if consecutive_recognitions[person_name] >= recognition_threshold_count:
                                if self.unlock_door(person_name):
                                    if not self.headless:
                                        cv2.putText(frame, f"UNLOCKED: {person_name}", 
                                                  (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.8, (0, 255, 0), 2)
                                else:
                                    if not self.headless:
                                        cv2.putText(frame, f"DOOR OPEN: {person_name}", 
                                                  (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.8, (0, 255, 255), 2)
                    else:
                        # No face detected - reset the access logged flag after a delay
                        if self.access_logged_for_current_session:
                            # Reset after 2 seconds of no detection
                            if current_time - self.last_access_log_time > 2:
                                self.access_logged_for_current_session = False
                else:
                    # Door is unlocked - don't process recognition, just show status
                    if not self.headless:
                        cv2.putText(frame, "DOOR UNLOCKED - Recognition Paused", 
                                  (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    # Clear consecutive recognitions while door is unlocked
                    consecutive_recognitions.clear()
                
                # Display system info (only if not headless)
                if not self.headless:
                    streaming_status = self.streaming_manager.get_streaming_status()
                    door_status = "LOCKED" if self.door_locked else "UNLOCKED"
                    cv2.putText(frame, f"Authorized Users: {len(self.authorized_faces)}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"API Users: {len(self.api_users)}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Door Status: {door_status}", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.door_locked else (0, 0, 255), 2)
                    cv2.putText(frame, "Face Recognition Door Lock", 
                              (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('Door Lock System', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    # In headless mode, just a small delay to prevent CPU overload
                    time.sleep(0.03)  # ~30 FPS equivalent
                    
        except KeyboardInterrupt:
            print("\nSystem stopped by user")
            self.speak("System shutting down.", priority=True)
            time.sleep(2)  # Give time for TTS to finish

    def check_door_lock_status(self, current_time):
        """Check if door should be locked again and reset access log status"""
        if not self.door_locked and current_time - self.last_unlock_time > self.unlock_duration:
            self.door_locked = True
            print("🔒 Door locked again - Recognition system reactivated")
            self.speak("Door locked. System ready for next user.")
            # Reset access log status when door locks
            self.access_logged_for_current_session = False

    def register_api_user(self, user_id):
        """Register face for a specific API user"""
        if user_id not in self.api_users:
            print(f"User ID {user_id} not found in API users")
            return False
        
        user_info = self.api_users[user_id]
        user_name = user_info['name']
        
        print(f"Registering face for API user: {user_name}")
        self.speak(f"Please register face for {user_name}")
        return self.add_authorized_face(user_name)

    def add_authorized_face(self, name):
        """Add a new authorized face to the system"""
        print(f"Adding authorized face for: {name}")
        self.speak(f"Registering face for {name}. Please look at the camera.")
        
        if not self.headless:
            print("Look at the camera and press SPACE when ready, or 'q' to cancel")
        else:
            print("Look at the camera. Face will be captured automatically when detected.")
        
        face_samples = []
        sample_count = 0
        required_samples = 20
        auto_capture_delay = 3.0  # seconds between auto captures in headless mode
        last_capture_time = 0
        
        while sample_count < required_samples:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces and get face encodings
            face_encoding = self.face_processor.process_frame_for_registration(frame, rgb_frame)
            
            current_time = time.time()
            
            if face_encoding is not None:
                if not self.headless:
                    # Show status on screen
                    cv2.putText(frame, f"Sample {sample_count + 1}/{required_samples}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Press SPACE to capture", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('Add Authorized Face', frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord(' '):
                        face_samples.append(face_encoding)
                        sample_count += 1
                        print(f"Captured sample {sample_count}/{required_samples}")
                        self.speak(f"Sample {sample_count} captured")
                    elif key == ord('q'):
                        print("Cancelled adding face")
                        self.speak("Face registration cancelled")
                        cv2.destroyWindow('Add Authorized Face')
                        return False
                else:
                    # Headless mode - auto capture with delay
                    if current_time - last_capture_time > auto_capture_delay:
                        face_samples.append(face_encoding)
                        sample_count += 1
                        last_capture_time = current_time
                        print(f"Auto-captured sample {sample_count}/{required_samples}")
                        self.speak(f"Sample {sample_count} captured")
            else:
                if self.headless and sample_count == 0:
                    # Provide guidance in headless mode
                    if current_time - last_capture_time > 3:  # Every 3 seconds
                        print("No face detected. Please position yourself in front of the camera.")
                        self.speak("Please position yourself in front of the camera")
                        last_capture_time = current_time
        
        if not self.headless:
            cv2.destroyWindow('Add Authorized Face')
        
        # Average the face samples
        if face_samples:
            avg_encoding = np.mean(face_samples, axis=0)
            self.authorized_faces[name] = avg_encoding
            self.save_authorized_faces()
            
            # Find user ID to update server
            user_id = None
            for uid, info in self.api_users.items():
                if info['name'] == name:
                    user_id = uid
                    break
            
            if user_id:
                if self.api_handler.send_registration_status(
                    user_id=user_id,
                    status=1,
                    face_encoding_data=avg_encoding
                ):
                    self.api_users[user_id]['status'] = 'registered'
                    self.save_api_users()  # Save updated status
                    print(f"Successfully registered {name} with server")
                    self.speak(f"Successfully registered {name}")
                else:
                    print(f"Warning: Failed to update server for {name}")
                    self.speak(f"Registration completed locally for {name}")
            
            print(f"Successfully added {name} to authorized faces")
            return True
        
        return False

    def process_access_attempt(self, frame, person_name, similarity):
        """Handle access attempt logging with proper user ID conversion - ONLY ONCE PER SESSION"""
        current_time = time.time()
        
        # Check if enough time has passed since last access log
        if current_time - self.last_access_log_time < self.access_log_cooldown:
            print(f"Access log cooldown active. Skipping log entry.")
            return False
        
        status = 1 if person_name else 2  # 1 for recognized, 2 for unrecognized
        
        # Determine user ID
        log_user_id = None
        if person_name:
            # Try to find user ID by matching name
            for uid, user_info in self.api_users.items():
                if user_info['name'] == person_name:
                    log_user_id = uid
                    break
            
            if log_user_id is None:
                log_user_id = -1  # Use -1 to indicate unknown registered user
        else:
            log_user_id = 0  # Use 0 for unrecognized faces
            self.speak("Access denied. Unauthorized person detected.")
        
        # Prepare detection data for streaming
        detection_data = {
            'person_name': person_name,
            'similarity': float(similarity) if similarity else None,
            'access_granted': bool(person_name),
            'user_id': log_user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add frame to streaming queue
        self.streaming_manager.add_frame(frame, detection_data)
        
        # Update streaming stats
        self.streaming_manager.update_stats(
            faces_detected=1 if person_name else 0,
            recognition_result=bool(person_name),
            additional_stats={
                'access_attempt': True,
                'access_status': status
            }
        )
        
        # Only log if we have a valid user ID or it's an unrecognized attempt
        if log_user_id is not None:
            print(f"Logging access attempt - User ID: {log_user_id}, Status: {status}")
            
            # Update last access log time
            self.last_access_log_time = current_time
            
            # Send the access log
            return self.api_handler.log_access(log_user_id, frame, status)
        else:
            print("Skipping access log due to invalid user ID")
            return False

    def get_user_id_by_name(self, name):
        """Helper method to get user ID by name"""
        for user_id, user_info in self.api_users.items():
            if user_info['name'] == name:
                return user_id
        return None

    def unlock_door(self, person_name):
        """Simulate door unlocking"""
        current_time = time.time()
        if current_time - self.last_unlock_time > self.unlock_duration:
            self.last_unlock_time = current_time
            self.door_locked = False  # Mark door as unlocked
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"🔓 DOOR UNLOCKED for {person_name} at {timestamp}")
            self.speak(f"Welcome {person_name}. Door unlocked.", priority=True)
            return True
        return False

    def save_authorized_faces(self):
        """Save authorized faces to file"""
        try:
            with open(self.face_encodings_file, 'wb') as f:
                pickle.dump(self.authorized_faces, f)
            print(f"Saved {len(self.authorized_faces)} authorized faces")
        except Exception as e:
            print(f"Error saving authorized faces: {e}")

    def load_authorized_faces(self):
        """Load authorized faces from file"""
        if os.path.exists(self.face_encodings_file):
            try:
                with open(self.face_encodings_file, 'rb') as f:
                    self.authorized_faces = pickle.load(f)
                print(f"Loaded {len(self.authorized_faces)} authorized faces")
            except Exception as e:
                print(f"Could not load authorized faces file: {e}")
                self.authorized_faces = {}
        else:
            print("No authorized faces file found")

    def save_api_users(self):
        """Save API users data to file for persistence"""
        try:
            with open(self.api_users_file, 'wb') as f:
                pickle.dump(self.api_users, f)
            print(f"Saved {len(self.api_users)} API users")
        except Exception as e:
            print(f"Error saving API users: {e}")

    def load_api_users(self):
        """Load API users from file"""
        if os.path.exists(self.api_users_file):
            try:
                with open(self.api_users_file, 'rb') as f:
                    self.api_users = pickle.load(f)
                print(f"Loaded {len(self.api_users)} API users from file")
                
                # Debug: Print loaded users
                for uid, info in self.api_users.items():
                    print(f"  - User ID {uid}: {info['name']}")
                    
            except Exception as e:
                print(f"Could not load API users file: {e}")
                self.api_users = {}
        else:
            print("No API users file found")

    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up resources...")
        self.streaming_manager.stop_streaming()
        
        # Final shutdown announcement
        if self.enable_tts:
            self.tts.stop_current_speech()
            self.tts.speak("System shutting down", priority=True)
            time.sleep(1)  # Give time for speech to complete
        
        self.cap.release()
        if not self.headless:
            cv2.destroyAllWindows()
        self.face_processor.cleanup()