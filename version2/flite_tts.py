import subprocess
import threading
import time
import os
import sys
from typing import Optional, List

class FliteTTS:
    """A Text-to-Speech manager using only Flite as the engine."""
    
    FLITE_VOICES = {
        'awb': 'Scottish male',
        'bdl': 'American male',
        'clb': 'American female',
        'jmk': 'German male',
        'ksp': 'Indian male',
        'rms': 'American male (rough)',
        'slt': 'American female (default)'
    }
    
    def __init__(self, voice: str = 'slt', speech_rate: int = 170, 
                 volume: float = 1.0, timeout: int = 30):
        """
        Initialize Flite TTS engine.
        
        Args:
            voice (str): Voice to use (default: 'slt')
            speech_rate (int): Speech rate in words per minute
            volume (float): Volume level (0.0 to 1.0)
            timeout (int): Timeout for speech operations in seconds
        """
        self.voice = voice
        self.speech_rate = speech_rate
        self.volume = volume
        self.timeout = timeout
        self.speaking = False
        self.speech_queue = []
        self.speech_thread = None
        self.stop_speaking = False
        
        # First try system path, then explicit path
        self.flite_path = self._find_flite()
        self.flite_available = self._check_flite_installation()
        
        if not self.flite_available:
            print("Flite TTS not available. Running in simulation mode.")
            print("To install Flite on Ubuntu/Debian: sudo apt-get install flite")
            print("To install Flite on CentOS/RHEL: sudo yum install flite")

    def _find_flite(self) -> str:
        """Find flite executable using multiple methods."""
        # Try which/where first
        try:
            result = subprocess.run(
                ['which', 'flite'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        # Common installation paths
        possible_paths = [
            '/usr/bin/flite',
            '/usr/local/bin/flite',
            '/opt/bin/flite',
            'flite'  # Last resort try system PATH
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        return 'flite'  # Fallback to hoping it's in PATH

    def _get_flite_search_paths(self) -> List[str]:
        """Get the paths that were searched for flite."""
        return [
            '/usr/bin/flite',
            '/usr/local/bin/flite',
            '/opt/bin/flite',
            os.environ.get('PATH', '')
        ]

    def _check_flite_installation(self) -> bool:
        """Check if Flite is installed and working."""
        try:
            # First check if the file exists and is executable
            if not os.path.exists(self.flite_path):
                print(f"Flite not found at: {self.flite_path}")
                return False
            
            if not os.access(self.flite_path, os.X_OK):
                print(f"Flite found but not executable: {self.flite_path}")
                return False
            
            # Try multiple ways to test flite
            test_methods = [
                # Method 1: Try --version (some versions don't support this)
                ([self.flite_path, '--version'], "version check"),
                # Method 2: Try -h or --help
                ([self.flite_path, '-h'], "help check"),
                # Method 3: Try with a simple test (most reliable)
                ([self.flite_path, '-t', 'test'], "simple speech test")
            ]
            
            for cmd, description in test_methods:
                try:
                    print(f"Testing Flite with {description}: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=5
                    )
                    
                    # For flite, return code might be non-zero even when working
                    # Check if we get any reasonable output or if it at least runs
                    if result.returncode == 0 or (result.stderr and b'flite' in result.stderr.lower()):
                        print(f"✓ Flite appears to be working (method: {description})")
                        if result.stdout:
                            print(f"  Output: {result.stdout.decode().strip()}")
                        return True
                    else:
                        print(f"  Return code: {result.returncode}")
                        if result.stdout:
                            print(f"  Stdout: {result.stdout.decode().strip()}")
                        if result.stderr:
                            print(f"  Stderr: {result.stderr.decode().strip()}")
                        
                except subprocess.TimeoutExpired:
                    print(f"  Timeout during {description}")
                except Exception as e:
                    print(f"  Exception during {description}: {e}")
            
            return False
            
        except Exception as e:
            print(f"Flite check exception: {str(e)}")
            return False

    def _start_speech_thread(self):
        """Start the speech thread if not already running."""
        if self.speech_thread is None or not self.speech_thread.is_alive():
            self.speech_thread = threading.Thread(target=self._speech_worker)
            self.speech_thread.daemon = True
            self.speech_thread.start()

    def _speech_worker(self):
        """Worker thread that processes the speech queue."""
        self.speaking = True
        
        while self.speech_queue and not self.stop_speaking:
            text = self.speech_queue.pop(0)
            if text and not self.stop_speaking:
                self._synthesize_speech(text)
        
        self.speaking = False

    def speak(self, text: str, priority: bool = False) -> None:
        """Add text to speech queue."""
        if not text or not text.strip():
            return
            
        if not self.flite_available:
            print(f"[SIMULATION] TTS would say: {text}")
            return

        if priority:
            self.stop_current_speech()
            self.speech_queue.insert(0, text)
        else:
            self.speech_queue.append(text)
        
        if not self.speaking:
            self._start_speech_thread()

    def _synthesize_speech(self, text: str) -> None:
        """Synthesize speech using Flite."""
        if self.stop_speaking or not self.flite_available:
            return
            
        try:
            # Basic flite command - keep it simple
            cmd = [self.flite_path, '-t', text]
            
            # Add voice if specified and available
            if self.voice and self.voice in self.FLITE_VOICES:
                cmd.extend(['-voice', self.voice])
            
            # Print the command for debugging
            print(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                print(f"Flite warning (return code {result.returncode}): {result.stderr.decode().strip()}")
                # Don't treat non-zero return as fatal error for flite
                
        except subprocess.TimeoutExpired:
            print("Flite speech timeout")
        except Exception as e:
            print(f"Flite speech error: {e}")

    def stop_current_speech(self) -> None:
        """Stop current speech."""
        self.stop_speaking = True
        
        try:
            # Try to kill any running flite processes
            subprocess.run(
                ['pkill', '-f', 'flite'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2
            )
        except Exception as e:
            print(f"Note: Could not kill flite processes: {e}")
        
        self.speech_queue.clear()
        time.sleep(0.5)
        self.stop_speaking = False

    def set_voice(self, voice: str) -> None:
        """Set the Flite voice to use."""
        if voice in self.FLITE_VOICES:
            self.voice = voice
            print(f"Voice set to: {voice} ({self.FLITE_VOICES[voice]})")
        else:
            print(f"Invalid voice. Available voices: {', '.join(self.FLITE_VOICES.keys())}")

    def set_speech_rate(self, rate: int) -> None:
        """Set speech rate in words per minute."""
        self.speech_rate = max(80, min(400, rate))  # Clamp to reasonable values

    def set_volume(self, volume: float) -> None:
        """Set volume level (0.0 to 1.0)."""
        self.volume = max(0.1, min(1.0, volume))  # Clamp to 0.1-1.0 range

    def get_available_voices(self) -> List[str]:
        """Get list of available Flite voices."""
        return list(self.FLITE_VOICES.keys())

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self.speaking

    def test_voice(self) -> None:
        """Test the current voice configuration."""
        test_text = "This is a test of the Flite text to speech system."
        print(f"Testing voice '{self.voice}' with text: {test_text}")
        self.speak(test_text, priority=True)

    def diagnose_flite(self) -> None:
        """Run comprehensive diagnostics on Flite installation."""
        print("=== Flite Diagnostics ===")
        print(f"Flite path: {self.flite_path}")
        print(f"File exists: {os.path.exists(self.flite_path)}")
        if os.path.exists(self.flite_path):
            print(f"File executable: {os.access(self.flite_path, os.X_OK)}")
            
        # Check what flite voices are actually available
        try:
            print("\nTesting available voices:")
            for voice in self.FLITE_VOICES.keys():
                try:
                    result = subprocess.run(
                        [self.flite_path, '-t', 'test', '-voice', voice],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=3
                    )
                    status = "✓ Available" if result.returncode == 0 else f"✗ Error (code {result.returncode})"
                    print(f"  {voice}: {status}")
                except Exception as e:
                    print(f"  {voice}: ✗ Exception: {e}")
        except Exception as e:
            print(f"Voice testing failed: {e}")
            
        # Try basic flite command
        try:
            print(f"\nTesting basic flite command:")
            result = subprocess.run(
                [self.flite_path, '-t', 'Hello world'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print(f"Stdout: {result.stdout.decode().strip()}")
            if result.stderr:
                print(f"Stderr: {result.stderr.decode().strip()}")
        except Exception as e:
            print(f"Basic test failed: {e}")

# Example usage
if __name__ == "__main__":
    try:
        print("Testing Flite TTS System")
        
        tts = FliteTTS(voice='slt')
        
        # Run diagnostics
        tts.diagnose_flite()
        
        if tts.flite_available:
            print("\n=== Testing TTS Functionality ===")
            print("Available voices:")
            for voice, desc in tts.FLITE_VOICES.items():
                print(f"  {voice}: {desc}")
            
            print("\nTesting default voice:")
            tts.test_voice()
            time.sleep(3)
            
            print("\nTesting queue system:")
            tts.speak("This is the first message in the queue.")
            tts.speak("This is the second message.")
            tts.speak("This is a high priority message.", priority=True)
            
            time.sleep(10)  # Wait for messages to complete
        else:
            print("\n=== Running in Simulation Mode ===")
            tts.speak("This would be spoken if Flite was working.")
            tts.speak("This is a second test message.")
        
    except Exception as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nStopping all speech...")
        if 'tts' in locals():
            tts.stop_current_speech()