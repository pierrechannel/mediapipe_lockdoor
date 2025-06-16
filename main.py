import cv2
import time
import threading
from datetime import datetime
from tts_manager import TTSManager
from face_recognition import FaceRecognition
from api_integration import APIIntegration
from door_lock import DoorLock

class FaceDoorLockSystem:
    def __init__(self, api_base_url, api_key=None, tts_engine='pyttsx3', tts_language='en'):
        # Initialize components
        self.tts = TTSManager(preferred_engine=tts_engine, language=tts_language)
        self.tts_enabled = True
        
        self.face_recognition = FaceRecognition()
        self.api = APIIntegration(api_base_url, api_key)
        self.door_lock = DoorLock()
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Settings
        self.last_speech_time = 0
        self.auto_registration_mode = False
        self.pending_registrations = []
        
        # Initial sync
        self.sync_users_from_api()
        
        print("Face Recognition Door Lock System Initialized")
        if self.tts_enabled:
            self.tts.speak("Face recognition door lock system initialized. Ready for access control.")

    def sync_users_from_api(self):
        """Fetch users from API and identify those needing registration"""
        if self.api.sync_users():
            # Identify users needing registration
            new_users = []
            for user_id in self.api.user_database:
                if user_id not in self.face_recognition.known_faces:
                    new_users.append(user_id)
            
            if new_users:
                print(f"Found {len(new_users)} users needing registration")
                self.pending_registrations = new_users

    def get_greeting_message(self, name, role):
        """Générer un message d'accueil approprié en français"""
        current_hour = datetime.now().hour
        
        # Salutation basée sur l'heure
        if 5 <= current_hour < 12:
            time_greeting = "Bonjour"
        elif 12 <= current_hour < 18:
            time_greeting = "Bon après-midi"
        else:
            time_greeting = "Bonsoir"
        
        # Salutation basée sur le rôle
        if role.upper() in ['ADMIN', 'ADMINISTRATOR', 'MANAGER']:
            return f"{time_greeting} {name}. Bienvenue, administrateur."
        elif role.upper() in ['SUPERVISOR', 'LEAD']:
            return f"{time_greeting} {name}. Accès autorisé, superviseur."
        else:
            return f"{time_greeting} {name}. Bienvenue dans l'entrepôt."

    def run(self):
        """Main application loop"""
        print("Face Recognition Door Lock Active")
        if self.tts_enabled:
            self.tts.speak("Face recognition door lock is now active. Looking for authorized personnel.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Recognize faces
                frame, recognized = self.face_recognition.recognize_face(frame)
                
                # Process recognized users
                for user_id, confidence in recognized:
                    user_info = self.api.get_user_info(user_id)
                    if user_info:
                        # Display user info
                        cv2.putText(frame, f"{user_info['name']} ({confidence:.2f})", 
                                   (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Role: {user_info['profile']}", 
                                   (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Announce and unlock
                        current_time = time.time()
                        if self.tts_enabled and current_time - self.last_speech_time > 5:
                            greeting = self.get_greeting_message(user_info['name'], user_info['profile'])
                            self.tts.speak(greeting)
                            self.last_speech_time = current_time
                        
                        # Unlock door
                        self.door_lock.unlock()
                        self.api.log_access(user_id, frame, status=1)
                
                # Display UI
                self._draw_ui(frame)
                cv2.imshow('Face Recognition Door Lock', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self._toggle_tts()
        
        finally:
            self.cleanup()

    def _draw_ui(self, frame):
        """Draw user interface elements on frame"""
        # Status information
        status = "LOCKED"
        status_color = (0, 0, 255)  # Red
        if time.time() - self.door_lock.last_unlock_time < self.door_lock.unlock_duration:
            status = "UNLOCKED"
            status_color = (0, 255, 0)  # Green
        
        cv2.putText(frame, f"Status: {status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # TTS status
        tts_status = "TTS: ON" if self.tts_enabled else "TTS: OFF"
        tts_color = (0, 255, 0) if self.tts_enabled else (0, 0, 255)
        cv2.putText(frame, tts_status, 
                   (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tts_color, 2)

    def _toggle_tts(self):
        """Toggle TTS on/off"""
        self.tts_enabled = not self.tts_enabled
        status = "enabled" if self.tts_enabled else "disabled"
        print(f"TTS {status}")
        
        if self.tts_enabled:
            self.tts.speak("Text to speech enabled.")
        else:
            self.tts.stop_speaking()

    def cleanup(self):
        """Clean up all resources"""
        print("Cleaning up system resources...")
        self.tts.cleanup()
        self.cap.release()
        self.door_lock.cleanup()
        cv2.destroyAllWindows()
        print("System shutdown complete")

if __name__ == "__main__":
    # Configuration
    API_BASE_URL = "https://apps.mediabox.bi:26875/"  # Replace with your actual API URL
    API_KEY = "your_api_key_here"  # Replace with your actual API key if required
    
    # TTS Configuration
    TTS_ENGINE = 'pyttsx3'  # Change to 'gtts' for better quality but requires internet
    TTS_LANGUAGE = 'en'     # Language code for TTS
    
    # Create and run the system
    system = FaceDoorLockSystem(
        api_base_url=API_BASE_URL,
        api_key=API_KEY,
        tts_engine=TTS_ENGINE,
        tts_language=TTS_LANGUAGE
    )
    system.run()