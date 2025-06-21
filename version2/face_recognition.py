import cv2
import numpy as np
import pickle
import os
from datetime import datetime
import time
import threading
import argparse
from face_processing import FaceProcessor
from api_handler import APIHandler
from streaming_manager import StreamingManager

# Text-to-speech imports
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("Warning: pyttsx3 not installed. Install with: pip install pyttsx3")
    TTS_AVAILABLE = False

class FaceRecognitionDoorLock:
    def __init__(self, api_base_url=None, api_headers=None, headless=False, enable_tts=True, enable_streaming=True):
        # Initialize components
        self.api_handler = APIHandler(api_base_url, api_headers)
        self.face_processor = FaceProcessor()
        
        # System configuration
        self.headless = headless
        self.enable_tts = enable_tts and TTS_AVAILABLE
        
        # Initialize Text-to-Speech
        if self.enable_tts:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)  # Speed of speech
                self.tts_engine.setProperty('volume', 0.8)  # Volume (0.0 to 1.0)
                
                # Get available voices and set a pleasant voice
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Try to find a female voice first, otherwise use the first available
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                    else:
                        self.tts_engine.setProperty('voice', voices[0].id)
                
                print("Text-to-speech initialized successfully")
            except Exception as e:
                print(f"Failed to initialize text-to-speech: {e}")
                self.enable_tts = False
        
        # Storage for authorized faces
        self.authorized_faces = {}
        self.api_users = {}  # Store users from API
        self.face_encodings_file = "authorized_faces.pkl"
        self.api_users_file = "api_users.pkl"  # Add persistence for API users
        
        # Load both authorized faces and API users
        self.load_authorized_faces()
        self.load_api_users()
        
        # Recognition parameters
        self.recognition_threshold = 0.6
        self.unlock_duration = 3  # seconds
        self.last_unlock_time = 0
        self.mode_check_interval = 10  # Check API every 10 seconds
        self.last_mode_check = 0
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        # TTS queue for threaded speech
        self.tts_queue = []
        self.tts_lock = threading.Lock()
        if self.enable_tts:
            self.start_tts_thread()
        self.enable_streaming = enable_streaming
        self.streaming_manager = StreamingManager(
        api_base_url=api_base_url,
        api_headers=api_headers,
        enable_streaming=enable_streaming
    )
    
    def start_system(self):
        """Start all system components including streaming"""
        # Start streaming service
        self.streaming_manager.start_streaming()
        print("Face recognition system with streaming started")
        if self.enable_tts:
            self.speak("System started with streaming enabled", priority=True)

    def speak(self, text, priority=False):
        """Add text to TTS queue or speak immediately if priority"""
        if not self.enable_tts:
            return
        
        with self.tts_lock:
            if priority:
                # Clear queue and speak immediately
                self.tts_queue.clear()
                self.tts_queue.append(text)
            else:
                self.tts_queue.append(text)

    def start_tts_thread(self):
        """Start TTS thread for non-blocking speech"""
        def tts_worker():
            while True:
                try:
                    with self.tts_lock:
                        if self.tts_queue:
                            text = self.tts_queue.pop(0)
                        else:
                            text = None
                    
                    if text:
                        print(f"🔊 Speaking: {text}")
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                    else:
                        time.sleep(0.1)  # Short sleep when queue is empty
                        
                except Exception as e:
                    print(f"TTS error: {e}")
                    time.sleep(1)
        
        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

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
                                if not self.headless:
                                    cv2.putText(frame, f"UNLOCKED: {person_name}", 
                                              (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.8, (0, 255, 0), 2)
                            else:
                                if not self.headless:
                                    cv2.putText(frame, f"DOOR OPEN: {person_name}", 
                                              (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.8, (0, 255, 255), 2)
                
                # Display system info (only if not headless)
                if not self.headless:
                    streaming_status = self.streaming_manager.get_streaming_status()
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
                else:
                    # In headless mode, just a small delay to prevent CPU overload
                    time.sleep(0.03)  # ~30 FPS equivalent
                    
        except KeyboardInterrupt:
            print("\nSystem stopped by user")
            self.speak("System shutting down.", priority=True)
            time.sleep(2)  # Give time for TTS to finish

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
        required_samples = 5
        auto_capture_delay = 1.0  # seconds between auto captures in headless mode
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
        """Handle access attempt logging with proper user ID conversion"""
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
        self.speak("System shutting down", priority=True)
        time.sleep(1)  # Give time for final TTS
        
        self.cap.release()
        if not self.headless:
            cv2.destroyAllWindows()
        self.face_processor.cleanup()
        
        if self.enable_tts:
            try:
                self.tts_engine.stop()
            except:
                pass

