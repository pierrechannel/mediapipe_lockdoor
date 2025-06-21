import cv2
import time
import threading
import numpy as np
import requests
import base64
import traceback
from datetime import datetime
from tts_manager import TTSManager
from face_recognition import FaceRecognition
from api_integration import APIIntegration
from door_lock import DoorLock
from live_streaming import StreamingIntegration

class FaceDoorLockSystem:
    def __init__(self, api_base_url, api_key=None, tts_engine='pyttsx3', tts_language='fr', 
                 streaming_port=8080, external_server_url=None, external_server_headers=None):
        try:
            # Initialize components
            self.tts = TTSManager(preferred_engine=tts_engine, language=tts_language)
            self.tts_enabled = True
            
            self.face_recognition = FaceRecognition()
            
            self.api = APIIntegration(api_base_url, api_key)
            
            self.door_lock = DoorLock()
            
            # Camera initialization with fallback
            self.cap = self._initialize_camera()
            
            # Registration tracking variables
            self.last_speech_time = 0
            self.registration_in_progress = False
            self.current_registration_user = None
            self.registration_announced = False
            self.registration_start_time = 0
            self.current_request_id = None
            self.current_user_info = None
            self.instruction_given = False
            
            # Registration monitoring control
            self.registration_check_interval = 10  # Check every 10 seconds instead of 5
            self.last_registration_check = 0
            self.registration_monitoring_active = True
            
            # External server configuration
            self.external_server_url = external_server_url
            self.external_server_headers = external_server_headers or {}
            self.post_stats = {
                'post_success_count': 0,
                'post_failure_count': 0
            }
            
            # Start components with priority on normal mode
            self._start_registration_monitoring()
            
            self.streaming_integration = StreamingIntegration(
                self, 
                streaming_port=streaming_port,
                external_server_url=external_server_url,
                external_server_headers=external_server_headers
            )
            self.streaming_integration.start_integrated_streaming()
            
            print("Face recognition door lock system initialized - Priority: Normal Mode")
            if self.tts_enabled:
                self.tts.speak("Système de reconnaissance faciale initialisé. Mode normal prioritaire.")
            
            if self.external_server_url:
                print(f"Configured to POST data to external server: {self.external_server_url}")
                
        except Exception as e:
            raise

    def _initialize_camera(self):
        """Initialize camera with fallback options"""
        try:
            # Try V4L2 first
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                raise RuntimeError("Could not open camera")
            
            # Set camera properties
            width_set = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            height_set = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            return cap
        except Exception as e:
            raise RuntimeError("Could not initialize camera")

    def _start_registration_monitoring(self):
        """Start the registration monitoring thread with lower priority"""
        try:
            monitor_thread = threading.Thread(
                target=self.monitor_registration,
                daemon=True,
                name="RegistrationMonitor"
            )
            # Set lower priority for registration monitoring
            monitor_thread.start()
            print("Registration monitoring started (lower priority)")
        except Exception as e:
            raise RuntimeError("Could not start registration monitoring")

    def monitor_registration(self):
        """Continuously check for registration requests with reduced frequency"""
        while self.registration_monitoring_active:
            try:
                current_time = time.time()
                
                # Only check for registration if not in progress and enough time has passed
                if (not self.registration_in_progress and 
                    current_time - self.last_registration_check >= self.registration_check_interval):
                    
                    self._check_for_new_registrations()
                    self.last_registration_check = current_time
                
                # Sleep longer to give priority to normal operations
                time.sleep(2)
                
            except Exception as e:
                time.sleep(5)

    def _check_for_new_registrations(self):
        """Check API for new registration requests - only when mode=2 and status=0"""
        try:
            response = self.api.get_system_mode_users()
            
            if response:
                if isinstance(response, dict):
                    status_code = response.get('statusCode')
                    
                    if status_code == 200:
                        result = response.get('result')
                        
                        # Check if result is null or empty - ignore registration mode
                        if result is None or result == {}:
                            print("API result is null/empty - staying in normal mode")
                            return
                        
                        if isinstance(result, dict):
                            mode = result.get('MODE')
                            status = result.get('STATUS')
                            
                            # Only process registration when mode=2 AND status=0
                            if mode == 2 and status == 0:
                                print(f"Registration request detected - Mode: {mode}, Status: {status}")
                                self._process_registration_request(result)
                            else:
                                # Log the current mode for debugging
                                if mode != 1:  # Only log if not in normal mode
                                    print(f"Current API mode: {mode}, status: {status} - Staying in normal mode")
                        else:
                            print(f"API result is not a dictionary - staying in normal mode (result type: {type(result)})")
                            
                    elif status_code == 404:
                        print("API endpoint not found (404) - staying in normal mode")
                        return
                        
                    else:
                        print(f"API returned status code {status_code} - staying in normal mode")
                        return
                else:
                    print("Invalid API response format - staying in normal mode")
            else:
                print("No response from API - staying in normal mode")
            
        except Exception as e:
            print(f"Error checking registrations: {e}")
            # Continue in normal mode even if there's an exception

    def _process_registration_request(self, request_data):
        """Process registration request with the new API format"""
        try:
            request_id = request_data.get('ID')
            user_id = request_data.get('WAREHOUSE_USER_ID')
            user_data = request_data.get('user', {})
            
            if not all([request_id, user_id, isinstance(user_data, dict)]):
                print("Invalid registration request data")
                return False
                
            prenom = user_data.get('PRENOM', '')
            nom = user_data.get('NOM', '')
            full_name = f"{prenom} {nom}".strip() or "Utilisateur"
            
            print(f"Processing registration for: {full_name} (ID: {user_id})")
            
            # Update registration status to "in progress"
            status_update_success = self.api.send_registration_status(user_id, 1)
            
            if not status_update_success:
                print("Failed to update registration status to 'in progress'")
                return False
                
            # Start registration mode
            registration_mode_started = self.face_recognition.start_registration_mode(user_id)
            
            if registration_mode_started:
                # Temporarily suspend normal registration checking to focus on current registration
                self.registration_monitoring_active = False
                
                # Set registration state
                self.registration_in_progress = True
                self.current_registration_user = user_id
                self.current_request_id = request_id
                self.registration_start_time = time.time()
                
                self.current_user_info = {
                    'request_id': request_id,
                    'user_id': user_id,
                    'name': full_name,
                    'email': user_data.get('EMAIL'),
                    'phone': user_data.get('TELEPHONE'),
                    'profile_id': user_data.get('PROFIL_ID', 1)
                }
                
                # TTS announcement
                if self.tts_enabled:
                    tts_message = f"Mode enregistrement activé pour {full_name}. Positionnez-vous devant la caméra."
                    self.tts.speak(tts_message)
                
                # Notify external server of registration start
                if self.external_server_url:
                    payload = {
                        'event': 'registration_start',
                        'user_id': user_id,
                        'name': full_name,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    }
                    self.post_to_server(payload=payload)
                
                print(f"Registration mode activated for {full_name}")
                return True
            else:
                print("Failed to start registration mode")
                return False
                
        except Exception as e:
            print(f"Error processing registration: {e}")
            if 'user_id' in locals():
                try:
                    self.api.send_registration_status(user_id, 2)  # Mark as failed
                except Exception:
                    pass
            return False

    def post_to_server(self, frame=None, payload=None):
        """Send data to an external server via POST request"""
        if not self.external_server_url:
            return False, "External server URL not configured"
        
        try:
            data = payload or {}
            if frame is not None:
                # Encode frame as JPEG and convert to base64
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
                    data['frame'] = frame_b64
                    data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            
            response = requests.post(
                self.external_server_url,
                json=data,
                headers=self.external_server_headers,
                timeout=5
            )
            
            if response.status_code == 200:
                self.post_stats['post_success_count'] += 1
                return True, "POST successful"
            else:
                self.post_stats['post_failure_count'] += 1
                return False, f"POST failed with status {response.status_code}"
                
        except requests.RequestException as e:
            self.post_stats['post_failure_count'] += 1
            return False, f"POST request failed: {str(e)}"

    def process_registration_completion(self, success, face_encoding=None):
        """Handle registration completion with face encoding support"""
        if not hasattr(self, 'current_user_info'):
            return False
            
        user_info = self.current_user_info
        status = 1 if success else 2
        
        print(f"Registration {'completed successfully' if success else 'failed'} for {user_info['name']}")
        
        # Prepare data for external server
        if self.external_server_url:
            payload = {
                'event': 'registration_complete',
                'user_id': user_info['user_id'],
                'name': user_info['name'],
                'status': 'success' if success else 'failed',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            }
            if face_encoding is not None and success:
                if isinstance(face_encoding, np.ndarray):
                    payload['face_encoding'] = face_encoding.tolist()
                else:
                    payload['face_encoding'] = face_encoding
            
            self.post_to_server(payload=payload)
        
        # Send status to API
        api_success = self.api.send_registration_status(
            user_id=user_info['user_id'],
            status=status,
            face_encoding_data=face_encoding if success else None
        )
        
        # User feedback
        if self.tts_enabled:
            if success:
                message = f"Enregistrement réussi pour {user_info['name']}. Retour au mode normal."
            else:
                message = f"Échec de l'enregistrement pour {user_info['name']}. Retour au mode normal."
            self.tts.speak(message)
        
        # Clean up and return to normal mode
        self._cleanup_registration()
        return api_success

    def _cleanup_registration(self):
        """Clean up registration state and return to normal mode priority"""
        print("Cleaning up registration state - returning to normal mode")
        
        self.registration_in_progress = False
        self.current_registration_user = None
        self.registration_announced = False
        self.registration_start_time = 0
        self.current_request_id = None
        
        # Resume normal registration monitoring
        self.registration_monitoring_active = True
        self.last_registration_check = 0  # Reset to check immediately on next cycle
        
        if hasattr(self, 'current_user_info'):
            delattr(self, 'current_user_info')
        if hasattr(self, 'instruction_given'):
            delattr(self, 'instruction_given')

    def get_greeting_message(self, name, role):
        """Generate appropriate greeting message in French"""
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:
            time_greeting = "Bonjour"
        elif 12 <= current_hour < 18:
            time_greeting = "Bon après-midi"
        else:
            time_greeting = "Bonsoir"
        
        if role.upper() in ['ADMIN', 'ADMINISTRATOR', 'MANAGER']:
            message = f"{time_greeting} {name}. Bienvenue, administrateur."
        elif role.upper() in ['SUPERVISOR', 'LEAD']:
            message = f"{time_greeting} {name}. Accès autorisé, superviseur."
        else:
            message = f"{time_greeting} {name}. Bienvenue dans nos locaux."
            
        return message
    def run(self):
        """Main application loop - prioritizes normal face recognition"""
        print("Système de reconnaissance faciale actif - Mode normal prioritaire")
        if self.tts_enabled:
            self.tts.speak("Système actif. Mode normal prioritaire pour le contrôle d'accès.")
        
        frame_count = 0
        recognition_attempts = 0
        last_unknown_log_time = 0  # Track last unknown person log to avoid spam
        unknown_log_cooldown = 10  # Log unknown person every 10 seconds max
        
        try:
            while True:
                frame_count += 1
                
                ret, frame = self.cap.read()
                if not ret:
                    print("Échec de la capture d'image")
                    time.sleep(1)
                    continue
                
                frame = cv2.flip(frame, 1)
                
                # Livestream every frame to the external server (reduced frequency during registration)
                if self.external_server_url:
                    # Stream less frequently during registration to focus on registration process
                    stream_frame = frame_count % (3 if self.registration_in_progress else 1) == 0
                    if stream_frame:
                        payload = {
                            'event': 'live_stream',
                            'frame_id': frame_count,
                            'mode': 'registration' if self.registration_in_progress else 'normal',
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                        }
                        self.post_to_server(frame=frame, payload=payload)
                
                # Face recognition processing
                recognition_attempts += 1
                recognized, registration_complete, face_encoding = self.face_recognition.recognize_face(frame)
            
                # Handle registration completion
                if registration_complete and self.registration_in_progress:
                    self.process_registration_completion(True, face_encoding)
                
                # Check for registration timeout (60 seconds)
                if (self.registration_in_progress and 
                    time.time() - self.registration_start_time > 60):
                    print("Registration timeout - cancelling and returning to normal mode")
                    if self.tts_enabled:
                        self.tts.speak("Délai d'enregistrement dépassé. Retour au mode normal.")
                    self.face_recognition.cancel_registration()
                    self.process_registration_completion(False)
                
                # Check for instruction prompt during registration
                if (self.registration_in_progress and 
                    self.face_recognition.registration_mode and
                    time.time() - self.registration_start_time > 10 and
                    self.face_recognition.registration_frames_collected < 3 and
                    not hasattr(self, 'instruction_given')):
                    
                    if self.tts_enabled:
                        self.tts.speak("Veuillez regarder directement la caméra et rester immobile.")
                        self.instruction_given = True

                # Process recognition results (PRIORITY - only when not in registration mode)
                if not self.registration_in_progress:
                    current_time = time.time()
                    
                    if recognized:
                        # Known person detected
                        for user_id, confidence in recognized:
                            user_info = self.api.get_user_info(user_id)
                            if user_info:
                                print(f"Access granted - User: {user_info['name']} (Confidence: {confidence:.2f})")
                                
                                time_since_last_speech = current_time - self.last_speech_time
                                
                                if self.tts_enabled and time_since_last_speech > 5:
                                    greeting = self.get_greeting_message(user_info['name'], user_info['profile'])
                                    self.tts.speak(greeting)
                                    self.last_speech_time = current_time
                                
                                self.door_lock.unlock()
                                # Log access for known user (status=1 for success)
                                self.api.log_access(user_id, frame, status=1)
                                
                                # Notify external server of successful recognition
                                if self.external_server_url:
                                    payload = {
                                        'event': 'access_granted',
                                        'user_id': user_id,
                                        'name': user_info['name'],
                                        'confidence': float(confidence),
                                        'role': user_info.get('profile', 'user'),
                                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                                    }
                                    self.post_to_server(frame=frame, payload=payload)
                    
                    else:
                        # Check if there's a face detected but not recognized (unknown person)
                        faces_detected = self.face_recognition.detect_faces(frame)
                        
                        if (faces_detected and 
                            current_time - last_unknown_log_time > unknown_log_cooldown):
                            
                            print("Unknown person detected - logging access attempt")
                            
                            # Log access for unknown user (status=0 for denied/unknown)
                            # Using user_id=-1 or 0 to indicate unknown person
                            self.api.log_access(-1, frame, status=0)
                            
                            # Notify external server of unknown person
                            if self.external_server_url:
                                payload = {
                                    'event': 'access_denied',
                                    'user_id': -1,
                                    'name': 'Unknown Person',
                                    'reason': 'face_not_recognized',
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                                }
                                self.post_to_server(frame=frame, payload=payload)
                            
                            # Optional: TTS notification for unknown person
                            if self.tts_enabled and current_time - self.last_speech_time > 10:
                                self.tts.speak("Accès refusé. Personne non reconnue.")
                                self.last_speech_time = current_time
                            
                            last_unknown_log_time = current_time
                
                # Shorter sleep during normal mode, longer during registration
                sleep_time = 0.05 if not self.registration_in_progress else 0.1
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nArrêt du système...")
            if self.tts_enabled:
                self.tts.speak("Arrêt du système en cours.")
        
        except Exception as e:
            print(f"Error in main loop: {e}")
            raise
    
    def cleanup(self):
        """Clean up all resources"""
        print("Nettoyage des ressources système...")
        
        try:
            # Stop registration monitoring
            self.registration_monitoring_active = False
            
            if self.face_recognition.registration_mode:
                self.face_recognition.cancel_registration()
            
            if hasattr(self.api, 'stop_registration_monitoring'):
                self.api.stop_registration_monitoring()
            
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            
            # Log system shutdown to external server
            if self.external_server_url:
                payload = {
                    'event': 'system_shutdown',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'post_stats': self.post_stats
                }
                self.post_to_server(payload=payload)
            
            cv2.destroyAllWindows()
            print("Arrêt complet du système")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    # Configuration
    API_BASE_URL = "https://apps.mediabox.bi:26875/"
    API_KEY = "your_api_key_here"
    TTS_ENGINE = 'pyttsx3'
    TTS_LANGUAGE = 'fr'
    EXTERNAL_SERVER_URL = "https://apps.mediabox.bi:26875/administration/streaming/receive_frame"
    EXTERNAL_SERVER_HEADERS = {'Authorization': 'Bearer your_token'}
    
    try:
        system = FaceDoorLockSystem(
            api_base_url=API_BASE_URL,
            api_key=API_KEY,
            tts_engine=TTS_ENGINE,
            tts_language=TTS_LANGUAGE,
            streaming_port=8080,
            external_server_url=EXTERNAL_SERVER_URL,
            external_server_headers=EXTERNAL_SERVER_HEADERS
        )
        system.run()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        raise