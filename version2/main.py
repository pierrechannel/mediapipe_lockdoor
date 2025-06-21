import os
import sys
import cv2
import time
import argparse
import logging
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
    
    # Optional dependencies
    optional = {
        'pyttsx3': ('pyttsx3', '2.90')  # For text-to-speech
    }
    
    missing = []
    outdated = []
    optional_missing = []
    
    # Check required dependencies
    for package, (pip_name, min_version) in required.items():
        try:
            mod = __import__(package)
            if hasattr(mod, '__version__'):
                try:
                    from pkg_resources import parse_version
                    if parse_version(mod.__version__) < parse_version(min_version):
                        outdated.append(f"{pip_name} (needs >= {min_version}, has {mod.__version__})")
                except:
                    # Fallback if pkg_resources not available
                    print(f"⚠️ Could not verify version for {pip_name}")
        except ImportError:
            missing.append(pip_name)
    
    # Check optional dependencies
    for package, (pip_name, min_version) in optional.items():
        try:
            __import__(package)
        except ImportError:
            optional_missing.append(pip_name)
    
    # Report status
    if missing or outdated:
        print("❌ System requirements check failed:")
        if missing:
            print("\n🔴 Missing required packages:")
            print("\n".join(f"  - {pkg}" for pkg in missing))
        if outdated:
            print("\n🟡 Outdated packages:")
            print("\n".join(f"  - {pkg}" for pkg in outdated))
        print("\n📦 Install/update with:")
        print(f"pip install --upgrade {' '.join(missing + [pkg.split()[0] for pkg in outdated])}")
        return False
    
    if optional_missing:
        print("⚠️ Optional dependencies missing:")
        print("\n".join(f"  - {pkg} (for enhanced features)" for pkg in optional_missing))
        print(f"\n💡 Install optional features with:")
        print(f"pip install {' '.join(optional_missing)}")
    
    return True

def load_config():
    """Load configuration with environment variable support"""
    config = {
        'API_BASE_URL': os.getenv('FACE_LOCK_API_URL', 'https://apps.mediabox.bi:26875'),
        'API_HEADERS': {
            'Content-Type': 'application/json',
        },
        'SYSTEM_SETTINGS': {
            'recognition_threshold': float(os.getenv('RECOGNITION_THRESHOLD', '0.65')),
            'unlock_duration': int(os.getenv('UNLOCK_DURATION', '3')),
            'mode_check_interval': int(os.getenv('MODE_CHECK_INTERVAL', '15'))
        }
    }
    
    # Add authorization token if provided
    api_token = os.getenv('FACE_LOCK_API_TOKEN')
    if api_token:
        config['API_HEADERS']['Authorization'] = f'Bearer {api_token}'
    
    return config

def setup_logging(debug_mode=False, headless=False):
    """Setup logging configuration"""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_file = f"logs/face_lock_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Setup file logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Console output
        ]
    )
    
    if headless:
        # In headless mode, ensure we have proper file logging
        logger = logging.getLogger()
        logger.info("Headless mode: All output will be logged to file")

def print_system_banner(headless=False, enable_tts=True):
    """Print system initialization banner"""
    banner = [
        "="*70,
        "🔐 FACE RECOGNITION DOOR LOCK SYSTEM",
        "="*70,
        f"🖥️  Mode: {'Headless' if headless else 'GUI'}",
        f"🔊 TTS: {'Enabled' if enable_tts else 'Disabled'}",
        f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*70
    ]
    
    for line in banner:
        print(line)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Face Recognition Door Lock System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with GUI and TTS
  python main.py --headless         # Run without GUI (server mode)
  python main.py --no-tts           # Run without text-to-speech
  python main.py --debug            # Enable debug logging
  python main.py --headless --debug # Headless with debug logging
        """
    )
    
    parser.add_argument('--headless', action='store_true',
                       help='Run without GUI display (headless mode)')
    parser.add_argument('--no-tts', action='store_true',
                       help='Disable text-to-speech functionality')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--api-url', type=str,
                       help='Override API base URL')
    parser.add_argument('--api-token', type=str,
                       help='API authentication token')
    parser.add_argument('--config-file', type=str,
                       help='Path to configuration file (not implemented)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug_mode=args.debug, headless=args.headless)
    logger = logging.getLogger(__name__)
    
    # Print system banner
    print_system_banner(headless=args.headless, enable_tts=not args.no_tts)
    
    # Check dependencies
    print("\n🔍 Checking system dependencies...")
    if not check_dependencies():
        print("\n❌ System startup aborted due to missing dependencies")
        return 1
    
    print("✅ All required dependencies satisfied")
    
    # Load configuration
    print("\n⚙️ Loading system configuration...")
    config = load_config()
    
    # Override with command line arguments
    if args.api_url:
        config['API_BASE_URL'] = args.api_url
        logger.info(f"API URL overridden: {args.api_url}")
    
    if args.api_token:
        config['API_HEADERS']['Authorization'] = f'Bearer {args.api_token}'
        logger.info("API token provided via command line")
    
    logger.info(f"Configuration loaded - API: {config['API_BASE_URL']}")
    
    try:
        # Initialize system with configuration
        print("\n🚀 Initializing system components...")
        door_lock = FaceRecognitionDoorLock(
            api_base_url=config['API_BASE_URL'],
            api_headers=config['API_HEADERS'],
            headless=args.headless,
            enable_tts=not args.no_tts
        )
        
        # Apply system settings
        door_lock.recognition_threshold = config['SYSTEM_SETTINGS']['recognition_threshold']
        door_lock.unlock_duration = config['SYSTEM_SETTINGS']['unlock_duration']
        door_lock.mode_check_interval = config['SYSTEM_SETTINGS']['mode_check_interval']
        
        logger.info("System components initialized successfully")
        
        # Initial synchronization
        print("\n🔄 Synchronizing with API server...")
        start_time = time.time()
        
        try:
            sync_result = door_lock.sync_api_users()
            sync_time = time.time() - start_time
            
            if sync_result:
                print(f"✅ Synchronization completed in {sync_time:.2f}s")
                print(f"📋 API users loaded: {len(door_lock.api_users)}")
                logger.info(f"API synchronization successful - {len(door_lock.api_users)} users")
            else:
                print(f"⚠️ Synchronization failed or no new users found ({sync_time:.2f}s)")
                print(f"📋 Using local face database: {len(door_lock.authorized_faces)} entries")
                logger.warning("API synchronization failed, using local database")
                
        except Exception as sync_error:
            print(f"⚠️ API synchronization error: {sync_error}")
            print(f"📋 Continuing with local face database: {len(door_lock.authorized_faces)} entries")
            logger.error(f"API sync error: {sync_error}")
        
        # Display system status
        print("\n" + "="*70)
        print("✅ SYSTEM READY - SECURITY ACTIVE")
        print("="*70)
        print(f"\n📊 System Configuration:")
        print(f"   🎯 Recognition threshold: {door_lock.recognition_threshold}")
        print(f"   ⏱️  Unlock duration: {door_lock.unlock_duration}s")
        print(f"   🔄 API check interval: {door_lock.mode_check_interval}s")
        print(f"   👥 Authorized faces: {len(door_lock.authorized_faces)}")
        print(f"   🌐 API users: {len(door_lock.api_users)}")
        
        if not args.headless:
            print(f"\n📹 Camera feed will open in new window...")
            print(f"🛑 Press 'Q' in camera window to shutdown system")
        else:
            print(f"\n🖥️  Running in headless mode (no display)")
            print(f"🛑 Press Ctrl+C to shutdown system")
        
        print()
        logger.info("System startup completed successfully")
        
        # Run main loop
        door_lock.run_door_lock_system()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Received shutdown signal from user")
        logger.info("System shutdown initiated by user")
    except Exception as e:
        print(f"\n❌ CRITICAL SYSTEM ERROR: {str(e)}")
        logger.critical(f"Critical system error: {str(e)}", exc_info=True)
        print("📋 Check logs for detailed error information")
        return 1
    finally:
        # Cleanup
        print("\n♻️ Shutting down system components...")
        try:
            if 'door_lock' in locals():
                door_lock.cleanup()
            logger.info("System cleanup completed")
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup warning: {cleanup_error}")
            logger.warning(f"Cleanup error: {cleanup_error}")
        
        print("\n" + "="*70)
        print("🛑 SYSTEM SHUTDOWN COMPLETE")
        print(f"📅 Stopped: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as fatal_error:
        print(f"\n💥 FATAL ERROR: {fatal_error}")
        logging.critical(f"Fatal error in main: {fatal_error}", exc_info=True)
        sys.exit(1)