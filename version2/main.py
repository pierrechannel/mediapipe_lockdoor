import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_recognition import FaceRecognitionDoorLock
from menu import Menu
import cv2
import mediapipe
import numpy
import pickle
import requests

def check_dependencies():
    """Check if required packages are available"""
    try:
        import mediapipe
        import cv2
        import numpy
        import pickle
        import requests
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Install required packages with:")
        print("pip install mediapipe opencv-python numpy requests")
        return False

def main():
    if not check_dependencies():
        return
    
    # Configure API settings
    API_BASE_URL = 'https://apps.mediabox.bi:26875'
    
    api_headers = None
    if API_BASE_URL:
        # You can configure headers here if needed (e.g., authentication tokens)
        api_headers = {
            'Content-Type': 'application/json',
            # 'Authorization': 'Bearer your-token-here'  # Add if needed
        }
    
    door_lock = FaceRecognitionDoorLock(api_base_url=API_BASE_URL, api_headers=api_headers)
    
    try:
        Menu.show_menu(door_lock)
    except KeyboardInterrupt:
        print("\nShutting down system...")
    finally:
        door_lock.cleanup()

if __name__ == "__main__":
    main()