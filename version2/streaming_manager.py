import cv2
import base64
import requests
import threading
import time
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import queue
import numpy as np
import os
import sys




class StreamingManager:
    """Manages streaming functionality for the face recognition door lock system"""
    
    def __init__(self, api_base_url: str, api_headers: Dict[str, str], enable_streaming: bool = True):
        self.api_base_url = api_base_url
        self.api_headers = api_headers
        self.enable_streaming = enable_streaming
        self.streaming_active = False
        self.stream_thread = None
        self.frame_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory issues
        self.stats_queue = queue.Queue(maxsize=50)
        self.logger = logging.getLogger(__name__)
        
        # Streaming configuration
        self.stream_config = {
            'frame_interval': 0.5,  # Send frame every 0.5 seconds
            'stats_interval': 2.0,  # Send stats every 2 seconds
            'max_retries': 3,
            'retry_delay': 1.0,
            'compression_quality': 80,  # JPEG compression quality (0-100)
            'frame_resize': (640, 480),  # Resize frames to reduce bandwidth
        }
        
        # Statistics tracking
        self.stats = {
            'frames_sent': 0,
            'frames_failed': 0,
            'stats_sent': 0,
            'stats_failed': 0,
            'last_frame_time': None,
            'last_stats_time': None,
            'connection_errors': 0,
            'total_faces_detected': 0,
            'total_recognitions': 0,
            'system_uptime_start': datetime.now()
        }

    def start_streaming(self):
        """Start the streaming service"""
        if not self.enable_streaming:
            self.logger.info("Streaming disabled in configuration")
            return
            
        if self.streaming_active:
            self.logger.warning("Streaming already active")
            return
            
        self.streaming_active = True
        self.stream_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self.stream_thread.start()
        self.logger.info("Streaming service started")

    def stop_streaming(self):
        """Stop the streaming service"""
        self.streaming_active = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5.0)
        self.logger.info("Streaming service stopped")

    def add_frame(self, frame: np.ndarray, detection_data: Optional[Dict] = None):
        """Add frame to streaming queue"""
        if not self.streaming_active:
            return
            
        try:
            # Resize frame if configured
            if self.stream_config['frame_resize']:
                frame = cv2.resize(frame, self.stream_config['frame_resize'])
            
            # Compress frame to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.stream_config['compression_quality']]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            frame_data = {
                'frame': frame_b64,
                'timestamp': datetime.now().isoformat(),
                'detection_data': detection_data or {}
            }
            
            # Add to queue (non-blocking)
            if not self.frame_queue.full():
                self.frame_queue.put(frame_data)
            else:
                # Remove oldest frame if queue is full
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame_data)
                except queue.Empty:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error adding frame to streaming queue: {e}")

    def update_stats(self, faces_detected: int = 0, recognition_result: bool = False, 
                    additional_stats: Optional[Dict] = None):
        """Update system statistics"""
        if not self.streaming_active:
            return
            
        try:
            current_time = datetime.now()
            uptime = (current_time - self.stats['system_uptime_start']).total_seconds()
            
            # Update counters
            if faces_detected > 0:
                self.stats['total_faces_detected'] += faces_detected
            if recognition_result:
                self.stats['total_recognitions'] += 1
            
            stats_data = {
                'timestamp': current_time.isoformat(),
                'system_uptime': uptime,
                'faces_detected_session': faces_detected,
                'total_faces_detected': self.stats['total_faces_detected'],
                'total_recognitions': self.stats['total_recognitions'],
                'frames_sent': self.stats['frames_sent'],
                'frames_failed': self.stats['frames_failed'],
                'stats_sent': self.stats['stats_sent'],
                'connection_errors': self.stats['connection_errors'],
                'recognition_success_rate': (
                    self.stats['total_recognitions'] / max(1, self.stats['total_faces_detected'])
                ) * 100,
                'last_frame_time': self.stats['last_frame_time'],
                'last_stats_time': self.stats['last_stats_time'],
            }
            
            # Add to stats queue
            if not self.stats_queue.full():
                self.stats_queue.put(stats_data)
            else:
                # Remove oldest stats if queue is full
                try:
                    self.stats_queue.get_nowait()
                    self.stats_queue.put(stats_data)
                except queue.Empty:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error updating streaming stats: {e}")

    def _streaming_worker(self):
        """Background worker for streaming data"""
        last_frame_time = 0
        last_stats_time = 0
        
        while self.streaming_active:
            try:
                current_time = time.time()
                
                # Send frames at configured interval
                if current_time - last_frame_time >= self.stream_config['frame_interval']:
                    self._send_frame()
                    last_frame_time = current_time
                
                # Send stats at configured interval
                if current_time - last_stats_time >= self.stream_config['stats_interval']:
                    self._send_stats()
                    last_stats_time = current_time
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in streaming worker: {e}")
                time.sleep(1.0)  # Wait before retrying

    def _send_frame(self):
        """Send frame data to server"""
        try:
            if self.frame_queue.empty():
                return
                
            frame_data = self.frame_queue.get_nowait()
            success = self._post_stream_data(frame=frame_data['frame'], 
                                           timestamp=frame_data['timestamp'],
                                           detection_data=frame_data['detection_data'])
            
            if success:
                self.stats['frames_sent'] += 1
                self.stats['last_frame_time'] = datetime.now()
            else:
                self.stats['frames_failed'] += 1
                
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error sending frame: {e}")
            self.stats['frames_failed'] += 1

    def _send_stats(self):
        """Send statistics data to server"""
        try:
            if self.stats_queue.empty():
                return
                
            stats_data = self.stats_queue.get_nowait()
            success = self._post_stream_data(stats=stats_data, 
                                           timestamp=stats_data['timestamp'])
            
            if success:
                self.stats['stats_sent'] += 1
                self.stats['last_stats_time'] = datetime.now()
            else:
                self.stats['stats_failed'] += 1
                
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error sending stats: {e}")
            self.stats['stats_failed'] += 1

    def _post_stream_data(self, frame: Optional[str] = None, 
                         timestamp: Optional[str] = None,
                         stats: Optional[Dict] = None,
                         detection_data: Optional[Dict] = None) -> bool:
        """Post streaming data to server with retry logic"""
        
        payload = {}
        if frame:
            payload['frame'] = frame
        if timestamp:
            payload['timestamp'] = timestamp
        if stats:
            payload['stats'] = stats
        if detection_data:
            payload['detection_data'] = detection_data
            
        if not payload:
            return False

        url = f"{self.api_base_url}/api/stream/receive"
        
        for attempt in range(self.stream_config['max_retries']):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=self.api_headers,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return True
                else:
                    self.logger.warning(f"Stream POST failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError as e:
                self.stats['connection_errors'] += 1
                self.logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
            except requests.exceptions.Timeout as e:
                self.logger.warning(f"Timeout error (attempt {attempt + 1}): {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error in stream POST (attempt {attempt + 1}): {e}")
            
            if attempt < self.stream_config['max_retries'] - 1:
                time.sleep(self.stream_config['retry_delay'])
        
        return False

    def get_streaming_status(self) -> Dict[str, Any]:
        """Get current streaming status and statistics"""
        return {
            'active': self.streaming_active,
            'enabled': self.enable_streaming,
            'config': self.stream_config,
            'stats': self.stats.copy(),
            'queue_status': {
                'frames_queued': self.frame_queue.qsize(),
                'stats_queued': self.stats_queue.qsize()
            }
        }


# Enhanced FaceRecognitionDoorLock class with streaming integration
class FaceRecognitionDoorLockWithStreaming:
    """Enhanced face recognition door lock with streaming capabilities"""
    
    def __init__(self, api_base_url: str, api_headers: Dict[str, str], 
                 headless: bool = False, enable_tts: bool = True, enable_streaming: bool = True):
        # Initialize base functionality (your existing FaceRecognitionDoorLock init code here)
        self.api_base_url = api_base_url
        self.api_headers = api_headers
        self.headless = headless
        self.enable_tts = enable_tts
        
        # Initialize streaming manager
        self.streaming_manager = StreamingManager(
            api_base_url=api_base_url,
            api_headers=api_headers,
            enable_streaming=enable_streaming
        )
        
        # Your existing attributes
        self.authorized_faces = []
        self.api_users = []
        self.recognition_threshold = 0.65
        self.unlock_duration = 3
        self.mode_check_interval = 15
        
        self.logger = logging.getLogger(__name__)

    def start_system(self):
        """Start the complete system including streaming"""
        # Start streaming service
        self.streaming_manager.start_streaming()
        self.logger.info("Face recognition system with streaming started")

    def process_camera_frame(self, frame: np.ndarray):
        """Process camera frame with streaming integration"""
        faces_detected = 0
        recognition_result = False
        detection_data = {}
        
        try:
            # Your existing face detection and recognition logic here
            # This is a placeholder - replace with your actual implementation
            
            # Example face detection logic
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            faces_detected = len(faces)
            
            if faces_detected > 0:
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Your face recognition logic here
                    # recognition_result = self.recognize_face(face_region)
                    
                detection_data = {
                    'faces_count': faces_detected,
                    'faces_locations': [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} 
                                      for (x, y, w, h) in faces],
                    'recognition_attempted': faces_detected > 0,
                    'access_granted': recognition_result
                }
            
            # Add frame to streaming queue
            self.streaming_manager.add_frame(frame, detection_data)
            
            # Update streaming stats
            self.streaming_manager.update_stats(
                faces_detected=faces_detected,
                recognition_result=recognition_result,
                additional_stats={
                    'frame_processing_time': time.time(),
                    'camera_active': True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing camera frame: {e}")
            
        return frame, faces_detected, recognition_result

    def run_door_lock_system(self):
        """Main system loop with streaming integration"""
        try:
            # Start streaming
            self.start_system()
            
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Cannot open camera")
            
            self.logger.info("Camera initialized, starting main loop")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to read frame from camera")
                    break
                
                # Process frame with streaming
                processed_frame, faces_detected, recognition_result = self.process_camera_frame(frame)
                
                # Display frame if not headless
                if not self.headless:
                    # Add streaming status overlay
                    streaming_status = self.streaming_manager.get_streaming_status()
                    status_text = f"Streaming: {'ON' if streaming_status['active'] else 'OFF'}"
                    stats_text = f"Frames: {streaming_status['stats']['frames_sent']}"
                    
                    cv2.putText(processed_frame, status_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(processed_frame, stats_text, (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Face Recognition Door Lock', processed_frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            self.logger.info("System interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in main system loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources including streaming"""
        try:
            # Stop streaming
            self.streaming_manager.stop_streaming()
            
            # Release camera
            if hasattr(self, 'cap'):
                self.cap.release()
            
            # Close windows
            cv2.destroyAllWindows()
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status including streaming"""
        streaming_status = self.streaming_manager.get_streaming_status()
        
        return {
            'door_lock_system': {
                'authorized_faces': len(self.authorized_faces),
                'api_users': len(self.api_users),
                'recognition_threshold': self.recognition_threshold,
                'headless_mode': self.headless,
                'tts_enabled': self.enable_tts
            },
            'streaming': streaming_status,
            'timestamp': datetime.now().isoformat()
        }