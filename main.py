import cv2
import time
import threading
import numpy as np
import requests
import base64
import logging
import traceback
from datetime import datetime
from tts_manager import TTSManager
from face_recognition import FaceRecognition
from api_integration import APIIntegration
from door_lock import DoorLock
from live_streaming import StreamingIntegration

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_door_lock_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FaceDoorLockSystem:
    def __init__(self, api_base_url, api_key=None, tts_engine='pyttsx3', tts_language='fr', 
                 streaming_port=8080, external_server_url=None, external_server_headers=None):
        logger.info("=== Initializing FaceDoorLockSystem ===")
        logger.info(f"API Base URL: {api_base_url}")
        logger.info(f"TTS Engine: {tts_engine}, Language: {tts_language}")
        logger.info(f"Streaming Port: {streaming_port}")
        logger.info(f"External Server URL: {external_server_url}")
        
        try:
            # Initialize components
            logger.debug("Initializing TTS Manager...")
            self.tts = TTSManager(preferred_engine=tts_engine, language=tts_language)
            self.tts_enabled = True
            logger.info("TTS Manager initialized successfully")
            
            logger.debug("Initializing Face Recognition...")
            self.face_recognition = FaceRecognition()
            logger.info("Face Recognition initialized successfully")
            
            logger.debug("Initializing API Integration...")
            self.api = APIIntegration(api_base_url, api_key)
            logger.info("API Integration initialized successfully")
            
            logger.debug("Initializing Door Lock...")
            self.door_lock = DoorLock()
            logger.info("Door Lock initialized successfully")
            
            # Camera initialization with fallback
            logger.debug("Initializing camera...")
            self.cap = self._initialize_camera()
            logger.info("Camera initialized successfully")
            
            # Registration tracking variables
            logger.debug("Setting up registration tracking variables...")
            self.last_speech_time = 0
            self.registration_in_progress = False
            self.current_registration_user = None
            self.registration_announced = False
            self.registration_start_time = 0
            self.current_request_id = None
            self.current_user_info = None
            self.instruction_given = False
            logger.debug("Registration tracking variables initialized")
            
            # External server configuration
            self.external_server_url = external_server_url
            self.external_server_headers = external_server_headers or {}
            self.post_stats = {
                'post_success_count': 0,
                'post_failure_count': 0
            }
            logger.info(f"External server configured: {bool(external_server_url)}")
            
            # Start components
            logger.debug("Starting registration monitoring...")
            self._start_registration_monitoring()
            
            logger.debug("Starting streaming integration...")
            self.streaming_integration = StreamingIntegration(
                self, 
                streaming_port=streaming_port,
                external_server_url=external_server_url,
                external_server_headers=external_server_headers
            )
            self.streaming_integration.start_integrated_streaming()
            logger.info("Streaming integration started successfully")
            
            logger.info("=== FaceDoorLockSystem initialization complete ===")
            print("Face recognition door lock system initialized")
            if self.tts_enabled:
                self.tts.speak("Système de reconnaissance faciale initialisé. Prêt pour le contrôle d'accès.")
            
            if self.external_server_url:
                logger.info(f"POST data configured for external server: {self.external_server_url}")
                print(f"Configured to POST data to external server: {self.external_server_url}")
                
        except Exception as e:
            logger.error(f"Failed to initialize FaceDoorLockSystem: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _initialize_camera(self):
        """Initialize camera with fallback options"""
        logger.debug("Attempting camera initialization...")
        try:
            # Try V4L2 first
            logger.debug("Trying V4L2 camera capture...")
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not cap.isOpened():
                logger.warning("V4L2 capture failed, trying default capture...")
                cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                logger.error("Both V4L2 and default capture failed")
                raise RuntimeError("Could not open camera")
            
            # Set camera properties
            logger.debug("Setting camera properties...")
            width_set = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            height_set = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            logger.debug(f"Width set: {width_set}, Height set: {height_set}")
            
            # Verify camera properties
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Camera initialized - Resolution: {actual_width}x{actual_height}, FPS: {fps}")
            
            return cap
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError("Could not initialize camera")

    def _start_registration_monitoring(self):
        """Start the registration monitoring thread"""
        logger.debug("Starting registration monitoring thread...")
        try:
            monitor_thread = threading.Thread(
                target=self.monitor_registration,
                daemon=True,
                name="RegistrationMonitor"
            )
            monitor_thread.start()
            logger.info(f"Registration monitoring thread started: {monitor_thread.name}")
            print("Registration monitoring started successfully")
        except Exception as e:
            logger.error(f"Failed to start registration monitoring: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError("Could not start registration monitoring")

    def monitor_registration(self):
        """Continuously check for registration requests"""
        logger.info("Registration monitoring loop started")
        iteration_count = 0
        while True:
            try:
                iteration_count += 1
                logger.debug(f"Registration monitoring iteration #{iteration_count}")
                
                if not self.registration_in_progress:
                    logger.debug("No registration in progress, checking for new registrations...")
                    self._check_for_new_registrations()
                else:
                    logger.debug(f"Registration in progress for user: {self.current_registration_user}")
                
                logger.debug("Registration monitoring sleeping for 5 seconds...")
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in registration monitoring iteration #{iteration_count}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(5)

    def _check_for_new_registrations(self):
        """Check API for new registration requests"""
        logger.debug("Checking API for new registration requests...")
        try:
            response = self.api.get_system_mode_users()
            logger.debug(f"API response received: {type(response)}")
            
            if response:
                logger.debug(f"API response content: {response}")
                
                if isinstance(response, dict):
                    status_code = response.get('statusCode')
                    logger.debug(f"Response status code: {status_code}")
                    
                    if status_code == 200:
                        result = response.get('result', {})
                        logger.debug(f"API result: {result}")
                        
                        if isinstance(result, dict):
                            mode = result.get('MODE')
                            status = result.get('STATUS')
                            logger.debug(f"Mode: {mode}, Status: {status}")
                            
                            if mode == 2 and status == 0:
                                logger.info("New registration request found!")
                                self._process_registration_request(result)
                            else:
                                logger.debug(f"No valid registration request (Mode: {mode}, Status: {status})")
                        else:
                            logger.warning(f"Result is not a dict: {type(result)}")
                    else:
                        logger.warning(f"API returned non-200 status: {status_code}")
                else:
                    logger.warning(f"Response is not a dict: {type(response)}")
            else:
                logger.debug("No response from API")
                
        except Exception as e:
            logger.error(f"Error checking for new registrations: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _process_registration_request(self, request_data):
        """Process registration request with the new API format"""
        logger.info("=== Processing Registration Request ===")
        logger.debug(f"Request data: {request_data}")
        
        try:
            request_id = request_data.get('ID')
            user_id = request_data.get('WAREHOUSE_USER_ID')
            user_data = request_data.get('user', {})
            
            logger.debug(f"Extracted - Request ID: {request_id}, User ID: {user_id}")
            logger.debug(f"User data: {user_data}")
            
            if not all([request_id, user_id, isinstance(user_data, dict)]):
                logger.error("Invalid registration request - missing required fields")
                logger.error(f"Request ID present: {bool(request_id)}")
                logger.error(f"User ID present: {bool(user_id)}")
                logger.error(f"User data is dict: {isinstance(user_data, dict)}")
                return False
                
            prenom = user_data.get('PRENOM', '')
            nom = user_data.get('NOM', '')
            full_name = f"{prenom} {nom}".strip() or "Utilisateur"
            logger.info(f"Processing registration for: {full_name} (ID: {user_id})")
            
            # Update registration status to "in progress"
            logger.debug("Sending registration status update (in progress)...")
            status_update_success = self.api.send_registration_status(user_id, 1)
            logger.debug(f"Status update success: {status_update_success}")
            
            if not status_update_success:
                logger.error("Failed to update registration status to 'in progress'")
                return False
                
            # Start registration mode
            logger.debug("Starting face recognition registration mode...")
            registration_mode_started = self.face_recognition.start_registration_mode(user_id)
            logger.debug(f"Registration mode started: {registration_mode_started}")
            
            if registration_mode_started:
                logger.info("Registration mode successfully started")
                
                # Set registration state
                self.registration_in_progress = True
                self.current_registration_user = user_id
                self.current_request_id = request_id
                self.registration_start_time = time.time()
                
                logger.debug(f"Registration state set - User: {user_id}, Request: {request_id}")
                
                self.current_user_info = {
                    'request_id': request_id,
                    'user_id': user_id,
                    'name': full_name,
                    'email': user_data.get('EMAIL'),
                    'phone': user_data.get('TELEPHONE'),
                    'profile_id': user_data.get('PROFIL_ID', 1)
                }
                logger.debug(f"User info stored: {self.current_user_info}")
                
                # TTS announcement
                if self.tts_enabled:
                    tts_message = f"Enregistrement facial pour {full_name}. Positionnez-vous devant la caméra."
                    logger.debug(f"TTS message: {tts_message}")
                    self.tts.speak(tts_message)
                
                # Notify external server of registration start
                if self.external_server_url:
                    logger.debug("Notifying external server of registration start...")
                    payload = {
                        'event': 'registration_start',
                        'user_id': user_id,
                        'name': full_name,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    }
                    success, message = self.post_to_server(payload=payload)
                    logger.debug(f"External server notification: success={success}, message={message}")
                    if not success:
                        logger.warning(f"Failed to notify external server of registration start: {message}")
                
                logger.info("=== Registration request processing complete - SUCCESS ===")
                return True
            else:
                logger.error("Failed to start registration mode")
                return False
                
        except Exception as e:
            logger.error(f"Error processing registration: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if 'user_id' in locals():
                logger.debug("Attempting to mark registration as failed...")
                try:
                    self.api.send_registration_status(user_id, 2)  # Mark as failed
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup registration status: {cleanup_error}")
            return False

    def post_to_server(self, frame=None, payload=None):
        """Send data to an external server via POST request"""
        if not self.external_server_url:
            logger.debug("No external server URL configured")
            return False, "External server URL not configured"
        
        logger.debug(f"Posting to server: {self.external_server_url}")
        logger.debug(f"Payload keys: {list(payload.keys()) if payload else 'None'}")
        logger.debug(f"Frame provided: {frame is not None}")
        
        try:
            data = payload or {}
            if frame is not None:
                logger.debug("Encoding frame to base64...")
                # Encode frame as JPEG and convert to base64
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
                    data['frame'] = frame_b64
                    data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    logger.debug(f"Frame encoded, size: {len(frame_b64)} characters")
                else:
                    logger.error("Failed to encode frame as JPEG")
            
            logger.debug(f"Sending POST request with headers: {self.external_server_headers}")
            response = requests.post(
                self.external_server_url,
                json=data,
                headers=self.external_server_headers,
                timeout=5
            )
            
            logger.debug(f"POST response status: {response.status_code}")
            logger.debug(f"POST response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                self.post_stats['post_success_count'] += 1
                logger.info(f"POST successful (Total successes: {self.post_stats['post_success_count']})")
                return True, "POST successful"
            else:
                self.post_stats['post_failure_count'] += 1
                logger.warning(f"POST failed with status {response.status_code} (Total failures: {self.post_stats['post_failure_count']})")
                logger.debug(f"Response content: {response.text[:500]}")  # First 500 chars
                return False, f"POST failed with status {response.status_code}"
                
        except requests.RequestException as e:
            self.post_stats['post_failure_count'] += 1
            logger.error(f"POST request failed: {str(e)} (Total failures: {self.post_stats['post_failure_count']})")
            return False, f"POST request failed: {str(e)}"

    def process_registration_completion(self, success, face_encoding=None):
        """Handle registration completion with face encoding support"""
        logger.info("=== Processing Registration Completion ===")
        logger.info(f"Success: {success}")
        logger.debug(f"Face encoding provided: {face_encoding is not None}")
        
        if not hasattr(self, 'current_user_info'):
            logger.error("No current user info available for registration completion")
            return False
            
        user_info = self.current_user_info
        status = 1 if success else 2
        logger.debug(f"User info: {user_info}")
        logger.debug(f"Registration status to send: {status}")
        
        # Prepare data for external server
        if self.external_server_url:
            logger.debug("Preparing external server notification...")
            payload = {
                'event': 'registration_complete',
                'user_id': user_info['user_id'],
                'name': user_info['name'],
                'status': 'success' if success else 'failed',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            }
            if face_encoding is not None and success:
                # Convert face encoding to list for JSON serialization
                if isinstance(face_encoding, np.ndarray):
                    payload['face_encoding'] = face_encoding.tolist()
                    logger.debug(f"Face encoding converted to list, length: {len(payload['face_encoding'])}")
                else:
                    payload['face_encoding'] = face_encoding
                    logger.debug(f"Face encoding already in correct format: {type(face_encoding)}")
            
            success_post, message = self.post_to_server(payload=payload)
            logger.debug(f"External server notification result: {success_post}, {message}")
            if not success_post:
                logger.warning(f"Failed to notify external server of registration completion: {message}")
        
        # Send status to API
        logger.debug("Sending registration status to API...")
        api_success = self.api.send_registration_status(
            user_id=user_info['user_id'],
            status=status,
            face_encoding_data=face_encoding if success else None
        )
        logger.debug(f"API status update success: {api_success}")
        
        # User feedback
        if self.tts_enabled:
            message = (f"Enregistrement réussi pour {user_info['name']}" 
                      if success else f"Échec de l'enregistrement pour {user_info['name']}")
            logger.debug(f"TTS feedback message: {message}")
            self.tts.speak(message)
        
        # Clean up
        logger.debug("Cleaning up registration state...")
        self._cleanup_registration()
        logger.info("=== Registration completion processing finished ===")
        return api_success

    def _cleanup_registration(self):
        """Clean up registration state"""
        logger.debug("Starting registration cleanup...")
        
        logger.debug(f"Before cleanup - Registration in progress: {self.registration_in_progress}")
        logger.debug(f"Before cleanup - Current user: {self.current_registration_user}")
        
        self.registration_in_progress = False
        self.current_registration_user = None
        self.registration_announced = False
        self.registration_start_time = 0
        self.current_request_id = None
        
        if hasattr(self, 'current_user_info'):
            logger.debug("Removing current_user_info attribute")
            delattr(self, 'current_user_info')
        if hasattr(self, 'instruction_given'):
            logger.debug("Removing instruction_given attribute")
            delattr(self, 'instruction_given')
            
        logger.info("Registration cleanup completed")

    def get_greeting_message(self, name, role):
        """Generate appropriate greeting message in French"""
        current_hour = datetime.now().hour
        logger.debug(f"Generating greeting for {name} with role {role} at hour {current_hour}")
        
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
            
        logger.debug(f"Generated greeting: {message}")
        return message

    def run(self):
        """Main application loop"""
        logger.info("=== Starting main application loop ===")
        print("Système de reconnaissance faciale actif")
        if self.tts_enabled:
            self.tts.speak("Système de reconnaissance faciale maintenant actif. Recherche de personnel autorisé.")
        
        frame_count = 0
        recognition_attempts = 0
        
        try:
            while True:
                frame_count += 1
                logger.debug(f"Processing frame #{frame_count}")
                
                ret, frame = self.cap.read()
                if not ret:
                    logger.error(f"Failed to capture frame #{frame_count}")
                    print("Échec de la capture d'image")
                    time.sleep(1)
                    continue
                
                logger.debug(f"Frame #{frame_count} captured successfully, shape: {frame.shape}")
                frame = cv2.flip(frame, 1)
                
                # Livestream every frame to the external server
                if self.external_server_url:
                    payload = {
                        'event': 'live_stream',
                        'frame_id': frame_count,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    }
                    success, message = self.post_to_server(frame=frame, payload=payload)
                    if not success:
                        logger.warning(f"Failed to stream frame #{frame_count}: {message}")
                    else:
                        logger.debug(f"Frame #{frame_count} streamed successfully")
                
                # Recognize faces (returns face encoding if registration complete)
                logger.debug("Starting face recognition...")
                recognition_attempts += 1
                recognized, registration_complete, face_encoding = self.face_recognition.recognize_face(frame)
            
                
                logger.debug(f"Recognition attempt #{recognition_attempts} results:")
                logger.debug(f"  - Recognized users: {len(recognized) if recognized else 0}")
                logger.debug(f"  - Registration complete: {registration_complete}")
                logger.debug(f"  - Face encoding provided: {face_encoding is not None}")
                
                if registration_complete and self.registration_in_progress:
                    logger.info("Registration completed successfully!")
                    self.process_registration_completion(True, face_encoding)
                
                # Check for registration timeout
                if (self.registration_in_progress and 
                    time.time() - self.registration_start_time > 60):
                    logger.warning("Registration timeout reached - cancelling registration")
                    print("Timeout d'enregistrement - annulation")
                    self.face_recognition.cancel_registration()
                    self.process_registration_completion(False)
                
                # Check for instruction prompt during registration
                if (self.registration_in_progress and 
                    self.face_recognition.registration_mode and
                    time.time() - self.registration_start_time > 10 and
                    self.face_recognition.registration_frames_collected < 3 and
                    not hasattr(self, 'instruction_given')):
                    
                    logger.info("Giving registration instruction to user")
                    if self.tts_enabled:
                        self.tts.speak("Veuillez regarder directement la caméra et rester immobile.")
                        self.instruction_given = True
                # registration_in_progress=False
                # Process recognized users (only when not in registration mode)
                if not self.registration_in_progress:
                    for user_id, confidence in recognized:
                        logger.info(f"Processing recognized user: {user_id} with confidence: {confidence:.2f}")
                        
                        user_info = self.api.get_user_info(user_id)
                        if user_info:
                            logger.info(f"User info retrieved: {user_info}")
                            print(f"Utilisateur reconnu: {user_info['name']} (Confiance: {confidence:.2f})")
                            
                            current_time = time.time()
                            time_since_last_speech = current_time - self.last_speech_time
                            logger.debug(f"Time since last speech: {time_since_last_speech:.2f}s")
                            
                            if self.tts_enabled and time_since_last_speech > 5:
                                greeting = self.get_greeting_message(user_info['name'], user_info['profile'])
                                logger.debug(f"Speaking greeting: {greeting}")
                                self.tts.speak(greeting)
                                self.last_speech_time = current_time
                            
                            logger.debug("Unlocking door...")
                            self.door_lock.unlock()
                            
                            logger.debug("Logging access to API...")
                            self.api.log_access(user_id, frame, status=1)
                            
                            # Notify external server of successful recognition
                            if self.external_server_url:
                                logger.debug("Notifying external server of access granted...")
                                payload = {
                                    'event': 'access_granted',
                                    'user_id': user_id,
                                    'name': user_info['name'],
                                    'confidence': float(confidence),
                                    'role': user_info.get('profile', 'user'),
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                                }
                                success, message = self.post_to_server(frame=frame, payload=payload)
                                if not success:
                                    logger.warning(f"Failed to notify external server of access: {message}")
                                    print(f"Failed to notify external server of access: {message}")
                                else:
                                    logger.info("Access granted and external server notified successfully")
                                    print("Access granted and external server notified successfully")
                        else:
                            logger.warning(f"Could not retrieve user info for user_id: {user_id}")

                
                logger.debug(f"Frame #{frame_count} processing complete, sleeping...")
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received - shutting down system")
            print("\nArrêt du système...")
            if self.tts_enabled:
                self.tts.speak("Arrêt du système en cours.")
        
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        finally:
            logger.info("Entering cleanup phase...")
            self.cleanup()

    def cleanup(self):
        """Clean up all resources"""
        logger.info("=== Starting system cleanup ===")
        print("Nettoyage des ressources système...")
        
        try:
            if self.face_recognition.registration_mode:
                logger.debug("Cancelling active registration...")
                self.face_recognition.cancel_registration()
            
            if hasattr(self.api, 'stop_registration_monitoring'):
                logger.debug("Stopping API registration monitoring...")
                self.api.stop_registration_monitoring()
            
            if hasattr(self, 'cap') and self.cap.isOpened():
                logger.debug("Releasing camera...")
                self.cap.release()
            
            # Log system shutdown to external server
            if self.external_server_url:
                logger.debug("Notifying external server of system shutdown...")
                payload = {
                    'event': 'system_shutdown',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'post_stats': self.post_stats
                }
                success, message = self.post_to_server(payload=payload)
                if not success:
                    logger.warning(f"Failed to notify external server of shutdown: {message}")
                    print(f"Failed to notify external server of shutdown: {message}")
                else:
                    logger.info("External server notified of shutdown successfully")
            
            logger.debug("Destroying OpenCV windows...")
            cv2.destroyAllWindows()
            
            logger.info("=== System cleanup completed ===")
            print("Arrêt complet du système")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    logger.info("=== Face Door Lock System Starting ===")
    
    # Configuration
     #API_BASE_URL = "http://127.0.0.1:5000/"
    API_BASE_URL = "https://apps.mediabox.bi:26875/"
    API_KEY = "your_api_key_here"
    TTS_ENGINE = 'pyttsx3'
    TTS_LANGUAGE = 'fr'
     #EXTERNAL_SERVER_URL = "http://127.0.0.1:5000/administration/streaming/receive_frame"  # Example external server
    EXTERNAL_SERVER_URL = "https://apps.mediabox.bi:26875/administration/streaming/receive_frame"  # Example external server
    EXTERNAL_SERVER_HEADERS = {'Authorization': 'Bearer your_token'}
    
    logger.info("Configuration loaded:")
    logger.info(f"  API_BASE_URL: {API_BASE_URL}")
    logger.info(f"  TTS_ENGINE: {TTS_ENGINE}")
    logger.info(f"  TTS_LANGUAGE: {TTS_LANGUAGE}")
    logger.info(f"  EXTERNAL_SERVER_URL: {EXTERNAL_SERVER_URL}")
    
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
        logger.info("System created successfully, starting main loop...")
        system.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"Fatal error: {e}")
        raise