import os
import sys
import cv2
import time
from datetime import datetime
from face_recognition import FaceRecognitionDoorLock

def check_dependencies():
    """Comprehensive dependency verification with version checks"""
    required = {
        'cv2': ('opencv-python', '4.5.5'),
        'mediapipe': ('mediapipe', '0.8.9'),
        'numpy': ('numpy', '1.21.0'),
        'requests': ('requests', '2.26.0')
    }
    
    missing = []
    outdated = []
    
    for package, (pip_name, min_version) in required.items():
        try:
            mod = __import__(package)
            if hasattr(mod, '__version__'):
                from pkg_resources import parse_version
                if parse_version(mod.__version__) < parse_version(min_version):
                    outdated.append(f"{pip_name} (needs >= {min_version}, has {mod.__version__})")
        except ImportError:
            missing.append(pip_name)
    
    if missing or outdated:
        print("System requirements check failed:")
        if missing:
            print("\nMissing packages:")
            print("\n".join(f"- {pkg}" for pkg in missing))
        if outdated:
            print("\nOutdated packages:")
            print("\n".join(f"- {pkg}" for pkg in outdated))
        print("\nInstall/update with:")
        print(f"pip install --upgrade {' '.join(missing + [pkg.split()[0] for pkg in outdated])}")
        return False
    return True

def main():
    # System initialization banner
    print("\n" + "="*60)
    print("🔐 FACE RECOGNITION DOOR LOCK SYSTEM INITIALIZATION")
    print("="*60)
    
    if not check_dependencies():
        print("\n❌ System startup aborted due to missing dependencies")
        return
    
    # Configuration - should be moved to config file in production
    CONFIG = {
        'API_BASE_URL': 'https://apps.mediabox.bi:26875',
        'API_HEADERS': {
            'Content-Type': 'application/json',
            # 'Authorization': 'Bearer YOUR_ACTUAL_TOKEN'
        },
        'SYSTEM_SETTINGS': {
            'recognition_threshold': 0.65,
            'unlock_duration': 3,
            'mode_check_interval': 15
        }
    }
    
    try:
        # Initialize system with configuration
        print("\n⚙️ Initializing system components...")
        door_lock = FaceRecognitionDoorLock(
            api_base_url=CONFIG['API_BASE_URL'],
            api_headers=CONFIG['API_HEADERS']
        )
        
        # Apply system settings
        door_lock.recognition_threshold = CONFIG['SYSTEM_SETTINGS']['recognition_threshold']
        door_lock.unlock_duration = CONFIG['SYSTEM_SETTINGS']['unlock_duration']
        door_lock.mode_check_interval = CONFIG['SYSTEM_SETTINGS']['mode_check_interval']
        
        # Initial synchronization
        print("\n🔄 Synchronizing with API server...")
        start_time = time.time()
        sync_result = door_lock.sync_api_users()
        sync_time = time.time() - start_time
        
        if sync_result:
            print(f"✅ Synchronization completed in {sync_time:.2f}s")
            print(f"📋 Registered API users: {len(door_lock.api_users)}")
        else:
            print(f"⚠️ Synchronization failed or no new users found ({sync_time:.2f}s)")
            print(f"📋 Using local face database: {len(door_lock.authorized_faces)} entries")
        
        # System ready message
        print("\n" + "="*60)
        print("✅ SYSTEM READY")
        print("="*60)
        print(f"\n🔹 Recognition threshold: {door_lock.recognition_threshold}")
        print(f"🔹 Unlock duration: {door_lock.unlock_duration}s")
        print(f"🔹 API check interval: {door_lock.mode_check_interval}s")
        print("\n📹 Camera feed activating...")
        print("🛑 Press 'Q' to shutdown system\n")
        
        # Run main loop
        door_lock.run_door_lock_system()
        
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal from user")
    except Exception as e:
        print(f"\n❌ CRITICAL SYSTEM ERROR: {str(e)}")
        print("Please check logs and configuration")
    finally:
        print("\n♻️ Shutting down system components...")
        door_lock.cleanup()
        print("\n" + "="*60)
        print("🛑 SYSTEM SHUTDOWN COMPLETE")
        print("="*60)

if __name__ == "__main__":
    # Add simple command line argument handling
    if '--help' in sys.argv or '-h' in sys.argv:
        print("Face Recognition Door Lock System")
        print("Usage: python main.py [--debug]")
        sys.exit(0)
        
    if '--debug' in sys.argv:
        print("🐛 DEBUG MODE ACTIVATED")
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    main()