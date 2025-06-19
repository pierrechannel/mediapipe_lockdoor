import cv2
import time
import threading
from datetime import datetime
from tts_manager import TTSManager
from face_recognition import FaceRecognition
from api_integration import APIIntegration
from door_lock import DoorLock

class FaceDoorLockSystem:
    def __init__(self, api_base_url, api_key=None, tts_engine='pyttsx3', tts_language='fr'):
        # Initialize components
        self.tts = TTSManager(preferred_engine=tts_engine, language=tts_language)
        self.tts_enabled = True
        
        self.face_recognition = FaceRecognition()
        self.api = APIIntegration(api_base_url, api_key)
        self.door_lock = DoorLock()
        
        # Camera - try different backends if needed
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)  # Fallback to default backend
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Settings
        self.last_speech_time = 0
        self.auto_registration_mode = False
        self.pending_registrations = []
        self.registration_in_progress = False
        self.current_registration_user = None
        
        # Registration announcement tracking
        self.registration_announced = False
        self.registration_start_time = 0
        
        # Initial sync
        self.sync_users_from_api()
        
        # Start registration monitoring
        self.start_registration_monitoring()
        
        print("Système de reconnaissance faciale initialisé")
        if self.tts_enabled:
            self.tts.speak("Système de reconnaissance faciale initialisé. Prêt pour le contrôle d'accès.")

    def sync_users_from_api(self):
        """Fetch users from API and identify those needing registration"""
        if self.api.sync_users():
            # Identify users needing registration
            new_users = []
            for user_id in self.api.user_database:
                if user_id not in self.face_recognition.known_faces:
                    new_users.append(user_id)
            
            if new_users:
                print(f"{len(new_users)} utilisateurs nécessitent un enregistrement")
                self.pending_registrations = new_users

    def start_registration_monitoring(self):
        """Démarre la surveillance des demandes d'enregistrement"""
        def monitor_registration():
            while True:
                try:
                    # Vérifier les demandes d'enregistrement depuis l'API
                    registration_requests = self.api.check_registration_requests()
                    
                    for request in registration_requests:
                        user_id = request.get('WAREHOUSE_USER_ID')
                        if user_id and not self.face_recognition.registration_mode:
                            self.handle_registration_request(user_id)
                    
                    time.sleep(5)  # Vérifier toutes les 5 secondes
                    
                except Exception as e:
                    print(f"Erreur dans la surveillance d'enregistrement: {e}")
                    time.sleep(5)
        
        # Démarrer le thread de surveillance
        monitor_thread = threading.Thread(target=monitor_registration, daemon=True)
        monitor_thread.start()
        print("Surveillance des demandes d'enregistrement démarrée")

    def handle_registration_request(self, user_id):
        """Gérer une demande d'enregistrement"""
        try:
            user_info = self.api.get_user_info(user_id)
            if user_info:
                print(f"Demande d'enregistrement reçue pour: {user_info['name']}")
                
                # Annoncer le début de l'enregistrement
                if self.tts_enabled:
                    message = f"Demande d'enregistrement pour {user_info['name']}. Veuillez vous positionner devant la caméra."
                    self.tts.speak(message)
                
                # Démarrer le mode d'enregistrement
                if self.face_recognition.start_registration_mode(user_id):
                    self.registration_in_progress = True
                    self.current_registration_user = user_id
                    self.registration_announced = True
                    self.registration_start_time = time.time()
                    
                    print(f"Mode d'enregistrement activé pour {user_info['name']}")
                    
        except Exception as e:
            print(f"Erreur lors du traitement de la demande d'enregistrement: {e}")
            self.api.send_registration_status(user_id, 2)  # Échec

    def process_registration_completion(self, success):
        """Traiter la fin d'un enregistrement"""
        if self.current_registration_user:
            user_info = self.api.get_user_info(self.current_registration_user)
            
            if success:
                print(f"Enregistrement réussi pour {user_info['name'] if user_info else self.current_registration_user}")
                if self.tts_enabled:
                    message = f"Enregistrement réussi pour {user_info['name'] if user_info else 'utilisateur'}. Bienvenue dans le système."
                    self.tts.speak(message)
                
                # Envoyer statut de succès à l'API
                self.api.send_registration_status(self.current_registration_user, 1)
                
            else:
                print(f"Échec de l'enregistrement pour {user_info['name'] if user_info else self.current_registration_user}")
                if self.tts_enabled:
                    message = f"Échec de l'enregistrement pour {user_info['name'] if user_info else 'utilisateur'}. Veuillez réessayer."
                    self.tts.speak(message)
                
                # Envoyer statut d'échec à l'API
                self.api.send_registration_status(self.current_registration_user, 2)
            
            # Réinitialiser les variables d'enregistrement
            self.registration_in_progress = False
            self.current_registration_user = None
            self.registration_announced = False
            self.registration_start_time = 0

    def get_greeting_message(self, name, role):
        """Generate appropriate greeting message in French"""
        current_hour = datetime.now().hour
        
        # Time-based greeting
        if 5 <= current_hour < 12:
            time_greeting = "Bonjour"
        elif 12 <= current_hour < 18:
            time_greeting = "Bon après-midi"
        else:
            time_greeting = "Bonsoir"
        
        # Role-based greeting
        if role.upper() in ['ADMIN', 'ADMINISTRATOR', 'MANAGER']:
            return f"{time_greeting} {name}. Bienvenue, administrateur."
        elif role.upper() in ['SUPERVISOR', 'LEAD']:
            return f"{time_greeting} {name}. Accès autorisé, superviseur."
        else:
            return f"{time_greeting} {name}. Bienvenue dans nos locaux."

    def run(self):
        """Main application loop - headless version"""
        print("Système de reconnaissance faciale actif")
        if self.tts_enabled:
            self.tts.speak("Système de reconnaissance faciale maintenant actif. Recherche de personnel autorisé.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Échec de la capture d'image")
                    time.sleep(1)  # Wait before retrying
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Recognize faces (gère aussi l'enregistrement)
                _, recognized, registration_complete = self.face_recognition.recognize_face(frame)
                
                # Vérifier si un enregistrement vient d'être terminé
                if registration_complete and self.registration_in_progress:
                    self.process_registration_completion(True)
                
                # Vérifier le timeout d'enregistrement (50 secondes)
                if (self.registration_in_progress and 
                    time.time() - self.registration_start_time > 50):
                    print("Timeout d'enregistrement - annulation")
                    self.face_recognition.cancel_registration()
                    self.process_registration_completion(False)
                
                # Donner des instructions pendant l'enregistrement
                if (self.registration_in_progress and 
                    self.face_recognition.registration_mode and
                    time.time() - self.registration_start_time > 10 and
                    self.face_recognition.registration_frames_collected < 3):
                    
                    if self.tts_enabled and not hasattr(self, 'instruction_given'):
                        self.tts.speak("Veuillez regarder directement la caméra et rester immobile.")
                        self.instruction_given = True
                
                # Process recognized users (seulement si pas en mode enregistrement)
                if not self.registration_in_progress:
                    for user_id, confidence in recognized:
                        user_info = self.api.get_user_info(user_id)
                        if user_info:
                            print(f"Utilisateur reconnu: {user_info['name']} (Confiance: {confidence:.2f})")
                            
                            # Announce and unlock
                            current_time = time.time()
                            if self.tts_enabled and current_time - self.last_speech_time > 5:
                                greeting = self.get_greeting_message(user_info['name'], user_info['profile'])
                                self.tts.speak(greeting)
                                self.last_speech_time = current_time
                            
                            # Unlock door
                            self.door_lock.unlock()
                            self.api.log_access(user_id, frame, status=1)
                
                # Add small delay to reduce CPU usage
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
        
        # Arrêter le mode d'enregistrement si actif
        if self.face_recognition.registration_mode:
            self.face_recognition.cancel_registration()
        
        # Arrêter la surveillance d'enregistrement
        self.api.stop_registration_monitoring()
        
        # Nettoyer les ressources
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("Arrêt complet du système")

if __name__ == "__main__":
    # Configuration
    #API_BASE_URL = "https://apps.mediabox.bi:26875/"  # Replace with your actual API URL
    API_BASE_URL = "http://127.0.0.1:5000/"  # Replace with your actual API URL

    API_KEY = "your_api_key_here"  # Replace with your actual API key if required
    
    # TTS Configuration - French language
    TTS_ENGINE = 'pyttsx3'  # Use 'gtts' for better quality but requires internet
    TTS_LANGUAGE = 'fr'     # French language code
    
    # Create and run the system
    system = FaceDoorLockSystem(
        api_base_url=API_BASE_URL,
        api_key=API_KEY,
        tts_engine=TTS_ENGINE,
        tts_language=TTS_LANGUAGE
    )
    system.run()