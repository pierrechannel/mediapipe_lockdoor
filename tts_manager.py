import os
import time
import threading
from io import BytesIO
import base64
from PIL import Image

# TTS imports
try:
    import pyttsx3
    TTS_PYTTSX3_AVAILABLE = True
    print("Moteur TTS pyttsx3 disponible")
except ImportError:
    TTS_PYTTSX3_AVAILABLE = False
    print("pyttsx3 non disponible - installez avec: pip install pyttsx3")

try:
    from gtts import gTTS
    import pygame
    TTS_GTTS_AVAILABLE = True
    print("Moteur gTTS disponible")
except ImportError:
    TTS_GTTS_AVAILABLE = False
    print("gTTS non disponible - installez avec: pip install gtts pygame")

class TTSManager:
    """Gère la fonctionnalité de synthèse vocale avec plusieurs moteurs"""
    
    def __init__(self, preferred_engine='pyttsx3', language='fr'):
        self.language = language  # Français par défaut
        self.preferred_engine = preferred_engine
        self.pyttsx3_engine = None
        self.audio_queue = []
        self.is_speaking = False
        
        # Initialiser le mixer pygame pour gTTS
        if TTS_GTTS_AVAILABLE:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Initialiser pyttsx3
        if TTS_PYTTSX3_AVAILABLE and preferred_engine == 'pyttsx3':
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self._configure_pyttsx3()
                print("Moteur pyttsx3 TTS initialisé")
            except Exception as e:
                print(f"Échec de l'initialisation de pyttsx3: {e}")
                self.pyttsx3_engine = None
        
        # Créer un répertoire temporaire pour les fichiers audio
        self.temp_dir = "temp_audio"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def _configure_pyttsx3(self):
        """Configurer pyttsx3 pour une meilleure qualité en français"""
        if not self.pyttsx3_engine:
            return
        
        try:
            # Obtenir les voix disponibles
            voices = self.pyttsx3_engine.getProperty('voices')
            
            # Essayer de trouver une voix française
            selected_voice = None
            for voice in voices:
                if 'french' in voice.languages or 'fr' in voice.languages:
                    selected_voice = voice.id
                    break
                elif 'français' in voice.name.lower():
                    selected_voice = voice.id
            
            if selected_voice:
                self.pyttsx3_engine.setProperty('voice', selected_voice)
            else:
                print("Aucune voix française trouvée - utilisation de la voix par défaut")
            
            # Configurer la vitesse de parole (mots par minute)
            self.pyttsx3_engine.setProperty('rate', 160)  # Vitesse normale pour le français
            
            # Configurer le volume (0.0 à 1.0)
            self.pyttsx3_engine.setProperty('volume', 0.9)
            
            print(f"TTS configuré - Voix: {selected_voice}, Vitesse: 160 MPM")
            
        except Exception as e:
            print(f"Erreur de configuration de pyttsx3: {e}")
    
    def speak_pyttsx3(self, text):
        """Parler en utilisant pyttsx3"""
        if not self.pyttsx3_engine:
            return False
        
        try:
            self.is_speaking = True
            self.pyttsx3_engine.say(text)
            self.pyttsx3_engine.runAndWait()
            self.is_speaking = False
            return True
        except Exception as e:
            print(f"Erreur de parole pyttsx3: {e}")
            self.is_speaking = False
            return False
    
    def speak_gtts(self, text):
        """Parler en utilisant Google TTS avec une meilleure qualité"""
        if not TTS_GTTS_AVAILABLE:
            return False
        
        try:
            self.is_speaking = True
            
            # Créer l'objet TTS en français
            tts = gTTS(text=text, lang=self.language, slow=False)
            
            # Sauvegarder dans un fichier temporaire
            temp_file = os.path.join(self.temp_dir, f"tts_{int(time.time())}.mp3")
            tts.save(temp_file)
            
            # Jouer le fichier audio
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Attendre la fin de la lecture
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Nettoyer le fichier temporaire
            try:
                os.remove(temp_file)
            except:
                pass
            
            self.is_speaking = False
            return True
            
        except Exception as e:
            print(f"Erreur de parole gTTS: {e}")
            self.is_speaking = False
            return False
    
    def speak(self, text, priority=False):
        """Méthode principale pour parler avec des moteurs de secours"""
        if not text or self.is_speaking:
            return
        
        print(f"TTS: {text}")
        
        def speak_worker():
            success = False
            
            # Essayer d'abord le moteur préféré
            if self.preferred_engine == 'pyttsx3' and TTS_PYTTSX3_AVAILABLE:
                success = self.speak_pyttsx3(text)
            elif self.preferred_engine == 'gtts' and TTS_GTTS_AVAILABLE:
                success = self.speak_gtts(text)
            
            # Revenir à d'autres moteurs si le préféré échoue
            if not success:
                if self.preferred_engine != 'pyttsx3' and TTS_PYTTSX3_AVAILABLE:
                    success = self.speak_pyttsx3(text)
                elif self.preferred_engine != 'gtts' and TTS_GTTS_AVAILABLE:
                    success = self.speak_gtts(text)
            
            if not success:
                print(f"Échec du TTS pour: {text}")
        
        # Exécuter le TTS dans un thread séparé pour ne pas bloquer
        tts_thread = threading.Thread(target=speak_worker, daemon=True)
        tts_thread.start()
    
    def stop_speaking(self):
        """Arrêter la parole en cours"""
        if TTS_GTTS_AVAILABLE:
            pygame.mixer.music.stop()
        
        if self.pyttsx3_engine:
            self.pyttsx3_engine.stop()
        
        self.is_speaking = False
    
    def cleanup(self):
        """Nettoyer les ressources TTS"""
        self.stop_speaking()
        
        # Nettoyer le répertoire temporaire
        try:
            for file in os.listdir(self.temp_dir):
                if file.endswith('.mp3'):
                    os.remove(os.path.join(self.temp_dir, file))
        except:
            pass