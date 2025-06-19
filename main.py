import cv2
import time
import threading
import numpy as np
import requests
import base64
from datetime import datetime
from tts_manager import TTSManager
from face_recognition import FaceRecognition
from api_integration import APIIntegration
from door_lock import DoorLock
from live_streaming import StreamingIntegration

class FaceDoorLockSystem:
    def __init__(self, api_base_url, api_key=None, tts_engine='pyttsx3', tts_language='fr', 
                 streaming_port=8080, external_server_url=None, external_server_headers=None):
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
        
        # External server configuration
        self.external_server_url = external_server_url
        self.external_server_headers = external_server_headers or {}
        self.post_stats = {
            'post_success_count': 0,
            'post_failure_count': 0
        }
        
        # Start components
        self._start_registration_monitoring()
        self.streaming_integration = StreamingIntegration(
            self, 
            streaming_port=streaming_port,
            external_server_url=external_server_url,
            external_server_headers=external_server_headers
        )
        self.streaming_integration.start_integrated_streaming()
        
        print("Face recognition door lock system initialized")
        if self.tts_enabled:
            self.tts.speak("Système de reconnaissance faciale initialisé. Prêt pour le contrôle d'accès.")
        
        if self.external_server_url:
            print(f"Configured to POST data to external server: {self.external_server_url}")

    def _initialize_camera(self):
        """Initialize camera with fallback options"""
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            raise RuntimeError("Could not initialize camera")

    def _start_registration_monitoring(self):
        """Start the registration monitoring thread"""
        try:
            monitor_thread = threading.Thread(
                target=self.monitor_registration,
                daemon=True,
                name="RegistrationMonitor"
            )
            monitor_thread.start()
            print("Registration monitoring started successfully")
        except Exception as e:
            print(f"Failed to start registration monitoring: {e}")
            raise RuntimeError("Could not start registration monitoring")

    def monitor_registration(self):
        """Continuously check for registration requests"""
        while True:
            try:
                if not self.registration_in_progress:
                    self._check_for_new_registrations()
                time.sleep(5)
            except Exception as e:
                print(f"Error in registration monitoring: {e}")
                time.sleep(5)

    def _check_for_new_registrations(self):
        """Check API for new registration requests"""
        try:
            response = self.api.get_system_mode_users()
            if response and isinstance(response, dict) and response.get('statusCode') == 200:
                result = response.get('result', {})
                if isinstance(result, dict) and result.get('MODE') == 2 and result.get('STATUS') == 0:
                    self._process_registration_request(result)
        except Exception as e:
            print(f"Error checking for new registrations: {e}")

    def _process_registration_request(self, request_data):
        """Process registration request with the new API format"""
        try:
            request_id = request_data.get('ID')
            user_id = request_data.get('WAREHOUSE_USER_ID')
            user_data = request_data.get('user', {})
            
            if not all([request_id, user_id, isinstance(user_data, dict)]):
                print("Invalid registration request - missing required fields")
                return False
                
            full_name = f"{user_data.get('PRENOM', '')} {user_data.get('NOM', '')}".strip() or "Utilisateur"
            
            if not self.api.send_registration_status(user_id, 1):  # Mark as in progress
                print("Failed to update registration status")
                return False
                
            if self.face_recognition.start_registration_mode(user_id):
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
                
                if self.tts_enabled:
                    self.tts.speak(f"Enregistrement facial pour {full_name}. Positionnez-vous devant la caméra.")
                
                # Notify external server of registration start
                if self.external_server_url:
                    payload = {
                        'event': 'registration_start',
                        'user_id': user_id,
                        'name': full_name,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    }
                    success, message = self.post_to_server(payload=payload)
                    if not success:
                        print(f"Failed to notify external server of registration start: {message}")
                
                return True
                
            return False
            
        except Exception as e:
            print(f"Error processing registration: {e}")
            if 'request_id' in locals():
                self.api.send_registration_status(user_id, 2)  # Mark as failed
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
                # Convert face encoding to list for JSON serialization
                payload['face_encoding'] = face_encoding.tolist() if isinstance(face_encoding, np.ndarray) else face_encoding
            
            success_post, message = self.post_to_server(payload=payload)
            if not success_post:
                print(f"Failed to notify external server of registration completion: {message}")
        
        # Send status to API
        api_success = self.api.send_registration_status(
            user_id=user_info['user_id'],
            status=status,
            face_encoding_data=face_encoding if success else None
        )
        
        # User feedback
        if self.tts_enabled:
            message = (f"Enregistrement réussi pour {user_info['name']}" 
                      if success else f"Échec de l'enregistrement pour {user_info['name']}")
            self.tts.speak(message)
        
        # Clean up
        self._cleanup_registration()
        return api_success

    def _cleanup_registration(self):
        """Clean up registration state"""
        self.registration_in_progress = False
        self.current_registration_user = None
        self.registration_announced = False
        self.registration_start_time = 0
        self.current_request_id = None
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
            return f"{time_greeting} {name}. Bienvenue, administrateur."
        elif role.upper() in ['SUPERVISOR', 'LEAD']:
            return f"{time_greeting} {name}. Accès autorisé, superviseur."
        else:
            return f"{time_greeting} {name}. Bienvenue dans nos locaux."

    def run(self):
        """Main application loop"""
        print("Système de reconnaissance faciale actif")
        if self.tts_enabled:
            self.tts.speak("Système de reconnaissance faciale maintenant actif. Recherche de personnel autorisé.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Échec de la capture d'image")
                    time.sleep(1)
                    continue
                
                frame = cv2.flip(frame, 1)
                
                # Recognize faces (returns face encoding if registration complete)
                recognized, registration_complete, face_encoding = self.face_recognition.recognize_face(frame)
                
                if registration_complete and self.registration_in_progress:
                    self.process_registration_completion(True, face_encoding)
                
                if (self.registration_in_progress and 
                    time.time() - self.registration_start_time > 50):
                    print("Timeout d'enregistrement - annulation")
                    self.face_recognition.cancel_registration()
                    self.process_registration_completion(False)
                
                if (self.registration_in_progress and 
                    self.face_recognition.registration_mode and
                    time.time() - self.registration_start_time > 10 and
                    self.face_recognition.registration_frames_collected < 3 and
                    not hasattr(self, 'instruction_given')):
                    
                    if self.tts_enabled:
                        self.tts.speak("Veuillez regarder directement la caméra et rester immobile.")
                        self.instruction_given = True
                
                if not self.registration_in_progress:
                    for user_id, confidence in recognized:
                        user_info = self.api.get_user_info(user_id)
                        if user_info:
                            print(f"Utilisateur reconnu: {user_info['name']} (Confiance: {confidence:.2f})")
                            
                            current_time = time.time()
                            if self.tts_enabled and current_time - self.last_speech_time > 5:
                                greeting = self.get_greeting_message(user_info['name'], user_info['profile'])
                                self.tts.speak(greeting)
                                self.last_speech_time = current_time
                            
                            self.door_lock.unlock()
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
                                success, message = self.post_to_server(frame=frame, payload=payload)
                                if not success:
                                    print(f"Failed to notify external server of access: {message}")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nArrêt du système...")
            if self.tts_enabled:
                self.tts.speak("Arrêt du système en cours.")
        
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up all resources"""
        print("Nettoyage des ressources système...")
        
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
            success, message = self.post_to_server(payload=payload)
            if not success:
                print(f"Failed to notify external server of shutdown: {message}")
        
        cv2.destroyAllWindows()
        print("Arrêt complet du système")

if __name__ == "__main__":
    # API_BASE_URL = "http://127.0.0.1:5000/"
    API_BASE_URL = "https://apps.mediabox.bi:26875/"
    API_KEY = "your_api_key_here"
    TTS_ENGINE = 'pyttsx3'
    TTS_LANGUAGE = 'fr'
    # EXTERNAL_SERVER_URL = "http://127.0.0.1:5000/administration/streaming/receive_frame"  # Example external server
    EXTERNAL_SERVER_URL = "https://apps.mediabox.bi:26875/administration/streaming/receive_frame"  # Example external server

    EXTERNAL_SERVER_HEADERS = {'Authorization': 'Bearer your_token'}
    
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