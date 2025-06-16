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
    print("pyttsx3 TTS engine available")
except ImportError:
    TTS_PYTTSX3_AVAILABLE = False
    print("pyttsx3 not available - install with: pip install pyttsx3")

try:
    from gtts import gTTS
    import pygame
    TTS_GTTS_AVAILABLE = True
    print("gTTS engine available")
except ImportError:
    TTS_GTTS_AVAILABLE = False
    print("gTTS not available - install with: pip install gtts pygame")

class TTSManager:
    """Handles Text-to-Speech functionality with multiple engines"""
    
    def __init__(self, preferred_engine='pyttsx3', language='en'):
        self.language = language
        self.preferred_engine = preferred_engine
        self.pyttsx3_engine = None
        self.audio_queue = []
        self.is_speaking = False
        
        # Initialize pygame mixer for gTTS playback
        if TTS_GTTS_AVAILABLE:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Initialize pyttsx3 engine
        if TTS_PYTTSX3_AVAILABLE and preferred_engine == 'pyttsx3':
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self._configure_pyttsx3()
                print("pyttsx3 TTS engine initialized")
            except Exception as e:
                print(f"Failed to initialize pyttsx3: {e}")
                self.pyttsx3_engine = None
        
        # Create temp directory for audio files
        self.temp_dir = "temp_audio"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def _configure_pyttsx3(self):
        """Configure pyttsx3 engine properties for better quality"""
        if not self.pyttsx3_engine:
            return
        
        try:
            # Get available voices
            voices = self.pyttsx3_engine.getProperty('voices')
            
            # Try to find a female voice or the best available voice
            selected_voice = None
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    selected_voice = voice.id
                    break
                elif 'english' in voice.name.lower():
                    selected_voice = voice.id
            
            if selected_voice:
                self.pyttsx3_engine.setProperty('voice', selected_voice)
            
            # Set speaking rate (words per minute)
            self.pyttsx3_engine.setProperty('rate', 180)  # Normal speaking rate
            
            # Set volume (0.0 to 1.0)
            self.pyttsx3_engine.setProperty('volume', 0.9)
            
            print(f"TTS configured - Voice: {selected_voice}, Rate: 180 WPM")
            
        except Exception as e:
            print(f"Error configuring pyttsx3: {e}")
    
    def speak_pyttsx3(self, text):
        """Speak using pyttsx3 engine"""
        if not self.pyttsx3_engine:
            return False
        
        try:
            self.is_speaking = True
            self.pyttsx3_engine.say(text)
            self.pyttsx3_engine.runAndWait()
            self.is_speaking = False
            return True
        except Exception as e:
            print(f"pyttsx3 speak error: {e}")
            self.is_speaking = False
            return False
    
    def speak_gtts(self, text):
        """Speak using Google TTS with better quality"""
        if not TTS_GTTS_AVAILABLE:
            return False
        
        try:
            self.is_speaking = True
            
            # Create TTS object
            tts = gTTS(text=text, lang=self.language, slow=False)
            
            # Save to temporary file
            temp_file = os.path.join(self.temp_dir, f"tts_{int(time.time())}.mp3")
            tts.save(temp_file)
            
            # Play the audio file
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
            
            self.is_speaking = False
            return True
            
        except Exception as e:
            print(f"gTTS speak error: {e}")
            self.is_speaking = False
            return False
    
    def speak(self, text, priority=False):
        """Main speak method with fallback engines"""
        if not text or self.is_speaking:
            return
        
        print(f"TTS: {text}")
        
        def speak_worker():
            success = False
            
            # Try preferred engine first
            if self.preferred_engine == 'pyttsx3' and TTS_PYTTSX3_AVAILABLE:
                success = self.speak_pyttsx3(text)
            elif self.preferred_engine == 'gtts' and TTS_GTTS_AVAILABLE:
                success = self.speak_gtts(text)
            
            # Fallback to other engine if preferred failed
            if not success:
                if self.preferred_engine != 'pyttsx3' and TTS_PYTTSX3_AVAILABLE:
                    success = self.speak_pyttsx3(text)
                elif self.preferred_engine != 'gtts' and TTS_GTTS_AVAILABLE:
                    success = self.speak_gtts(text)
            
            if not success:
                print(f"TTS failed for: {text}")
        
        # Run TTS in separate thread to avoid blocking
        tts_thread = threading.Thread(target=speak_worker, daemon=True)
        tts_thread.start()
    
    def stop_speaking(self):
        """Stop current speech"""
        if TTS_GTTS_AVAILABLE:
            pygame.mixer.music.stop()
        
        if self.pyttsx3_engine:
            self.pyttsx3_engine.stop()
        
        self.is_speaking = False
    
    def cleanup(self):
        """Clean up TTS resources"""
        self.stop_speaking()
        
        # Clean up temp directory
        try:
            for file in os.listdir(self.temp_dir):
                if file.endswith('.mp3'):
                    os.remove(os.path.join(self.temp_dir, file))
        except:
            pass