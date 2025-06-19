
import cv2
import threading
import time
from flask import Flask, Response, render_template_string, jsonify, request
import numpy as np
from datetime import datetime
import base64
import json
import requests  # Added for HTTP POST requests


class LiveStreaming:
    def __init__(self, host='0.0.0.0', port=8080, quality=85, fps_limit=15, external_server_url=None, external_server_headers=None):
        """
        Initialize live streaming server
        
        Args:
            host (str): Host address for streaming server
            port (int): Port for streaming server
            quality (int): JPEG quality (1-100)
            fps_limit (int): Maximum FPS for streaming
            external_server_url (str): URL of the external server to POST data to
            external_server_headers (dict): Optional headers for external server requests
        """
        self.host = host
        self.port = port
        self.quality = quality
        self.fps_limit = fps_limit
        self.frame_interval = 1.0 / fps_limit
        self.external_server_url = external_server_url
        self.external_server_headers = external_server_headers or {}
        
        # Flask app for streaming
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'face_recognition_streaming'
        
        # Streaming state
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.streaming_active = False
        self.last_frame_time = 0
        
        # Statistics
        self.stats = {
            'start_time': None,
            'frames_streamed': 0,
            'current_fps': 0,
            'active_connections': 0,
            'last_update': None,
            'post_success_count': 0,  # Track successful POSTs
            'post_failure_count': 0   # Track failed POSTs
        }
        
        # Setup Flask routes
        self.setup_routes()
        
        print(f"Live streaming server initialized on {host}:{port}")
        if self.external_server_url:
            print(f"Configured to POST data to external server: {self.external_server_url}")

    def setup_routes(self):
        """Setup Flask routes for streaming"""
        
        @self.app.route('/')
        def index():
            """Main streaming page"""
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Reconnaissance Faciale - Live Stream</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f0f0f0;
                    }
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                        background-color: white;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }
                    .header {
                        text-align: center;
                        margin-bottom: 20px;
                        color: #333;
                    }
                    .stream-container {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 20px;
                        align-items: flex-start;
                    }
                    .video-section {
                        flex: 2;
                        min-width: 300px;
                    }
                    .info-section {
                        flex: 1;
                        min-width: 250px;
                        background-color: #f8f9fa;
                        padding: 15px;
                        border-radius: 8px;
                    }
                    #stream {
                        width: 100%;
                        height: auto;
                        border: 2px solid #ddd;
                        border-radius: 8px;
                        background-color: #000;
                    }
                    .status {
                        display: inline-block;
                        padding: 5px 10px;
                        border-radius: 15px;
                        color: white;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }
                    .status.active { background-color: #28a745; }
                    .status.inactive { background-color: #dc3545; }
                    .stats-table {
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 10px;
                    }
                    .stats-table th, .stats-table td {
                        padding: 8px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }
                    .stats-table th {
                        background-color: #e9ecef;
                        font-weight: bold;
                    }
                    .controls {
                        margin-top: 15px;
                        text-align: center;
                    }
                    .btn {
                        padding: 8px 16px;
                        margin: 5px;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 14px;
                    }
                    .btn-primary { background-color: #007bff; color: white; }
                    .btn-secondary { background-color: #6c757d; color: white; }
                    .btn:hover { opacity: 0.8; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🎥 Système de Reconnaissance Faciale</h1>
                        <h2>Flux Vidéo en Direct</h2>
                        <div id="status" class="status inactive">Déconnecté</div>
                    </div>
                    
                    <div class="stream-container">
                        <div class="video-section">
                            <img id="stream" src="/video_feed" alt="Stream vidéo indisponible">
                            <div class="controls">
                                <button class="btn btn-primary" onclick="refreshStream()">🔄 Actualiser</button>
                                <button class="btn btn-secondary" onclick="toggleFullscreen()">🔍 Plein écran</button>
                            </div>
                        </div>
                        
                        <div class="info-section">
                            <h3>📊 Statistiques du Stream</h3>
                            <table class="stats-table" id="statsTable">
                                <tr><th>Statut</th><td id="streamStatus">-</td></tr>
                                <tr><th>FPS Actuel</th><td id="currentFps">-</td></tr>
                                <tr><th>Connexions</th><td id="connections">-</td></tr>
                                <tr><th>Images diffusées</th><td id="framesStreamed">-</td></tr>
                                <tr><th>Temps d'activité</th><td id="uptime">-</td></tr>
                                <tr><th>Dernière MAJ</th><td id="lastUpdate">-</td></tr>
                                <tr><th>POST Réussis</th><td id="postSuccess">-</td></tr>
                                <tr><th>POST Échoués</th><td id="postFailure">-</td></tr>
                            </table>
                            
                            <h3>ℹ️ Informations</h3>
                            <p><strong>Qualité:</strong> {{ quality }}%</p>
                            <p><strong>FPS Max:</strong> {{ fps_limit }}</p>
                            <p><strong>Serveur:</strong> {{ host }}:{{ port }}</p>
                            <p><strong>Serveur Externe:</strong> {{ external_server_url or 'Non configuré' }}</p>
                        </div>
                    </div>
                </div>
                
                <script>
                    let streamActive = false;
                    
                    // Check stream status
                    function updateStats() {
                        fetch('/stats')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('streamStatus').textContent = data.streaming_active ? 'Actif' : 'Inactif';
                                document.getElementById('currentFps').textContent = data.current_fps.toFixed(1);
                                document.getElementById('connections').textContent = data.active_connections;
                                document.getElementById('framesStreamed').textContent = data.frames_streamed;
                                document.getElementById('uptime').textContent = data.uptime || '-';
                                document.getElementById('lastUpdate').textContent = data.last_update || '-';
                                document.getElementById('postSuccess').textContent = data.post_success_count;
                                document.getElementById('postFailure').textContent = data.post_failure_count;
                                
                                const statusElement = document.getElementById('status');
                                if (data.streaming_active) {
                                    statusElement.textContent = 'Connecté';
                                    statusElement.className = 'status active';
                                } else {
                                    statusElement.textContent = 'Déconnecté';
                                    statusElement.className = 'status inactive';
                                }
                            })
                            .catch(error => {
                                console.error('Erreur lors de la récupération des stats:', error);
                            });
                    }
                    
                    function refreshStream() {
                        const img = document.getElementById('stream');
                        img.src = '/video_feed?' + new Date().getTime();
                    }
                    
                    function toggleFullscreen() {
                        const img = document.getElementById('stream');
                        if (img.requestFullscreen) {
                            img.requestFullscreen();
                        }
                    }
                    
                    // Update stats every 2 seconds
                    setInterval(updateStats, 2000);
                    updateStats();
                    
                    // Handle stream load events
                    document.getElementById('stream').onload = function() {
                        streamActive = true;
                    };
                    
                    document.getElementById('stream').onerror = function() {
                        streamActive = false;
                        setTimeout(refreshStream, 5000); // Retry after 5 seconds
                    };
                </script>
            </body>
            </html>
            """
            return render_template_string(html_template, 
                                        quality=self.quality,
                                        fps_limit=self.fps_limit,
                                        host=self.host,
                                        port=self.port,
                                        external_server_url=self.external_server_url)
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route"""
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/stats')
        def stats():
            """Statistics API endpoint"""
            current_time = time.time()
            uptime = None
            if self.stats['start_time']:
                uptime_seconds = current_time - self.stats['start_time']
                hours = int(uptime_seconds // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                seconds = int(uptime_seconds % 60)
                uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            return jsonify({
                'streaming_active': self.streaming_active,
                'current_fps': self.stats['current_fps'],
                'active_connections': self.stats['active_connections'],
                'frames_streamed': self.stats['frames_streamed'],
                'uptime': uptime,
                'last_update': self.stats['last_update'],
                'post_success_count': self.stats['post_success_count'],
                'post_failure_count': self.stats['post_failure_count']
            })

        @self.app.route('/start_streaming', methods=['POST'])
        def start_streaming():
            """Start streaming with configuration via POST request"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No JSON data provided'}), 400
                
                # Validate and update quality
                quality = data.get('quality', self.quality)
                if not isinstance(quality, int) or quality < 1 or quality > 100:
                    return jsonify({'error': 'Quality must be an integer between 1 and 100'}), 400
                self.quality = quality
                
                # Validate and update fps_limit
                fps_limit = data.get('fps_limit', self.fps_limit)
                if not isinstance(fps_limit, int) or fps_limit < 1:
                    return jsonify({'error': 'FPS limit must be a positive integer'}), 400
                self.fps_limit = fps_limit
                self.frame_interval = 1.0 / fps_limit
                
                return jsonify({
                    'status': 'Streaming configuration updated',
                    'quality': self.quality,
                    'fps_limit': self.fps_limit,
                    'stream_url': self.get_stream_url(),
                    'video_feed_url': self.get_video_feed_url()
                }), 200
            except Exception as e:
                return jsonify({'error': f'Failed to update streaming configuration: {str(e)}'}), 500

    def post_to_server(self, frame=None, stats=None):
        """Send frame or statistics to an external server via POST request"""
        if not self.external_server_url:
            return False, "External server URL not configured"
        
        try:
            payload = {}
            if frame is not None:
                # Encode frame as JPEG and convert to base64
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if ret:
                    frame_bytes = buffer.tobytes()
                    frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
                    payload['frame'] = frame_b64
                    payload['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            
            if stats is not None:
                payload['stats'] = stats
            
            response = requests.post(self.external_server_url, 
                                  json=payload, 
                                  headers=self.external_server_headers,
                                  timeout=5)
            
            if response.status_code == 200:
                self.stats['post_success_count'] += 1
                return True, "POST successful"
            else:
                self.stats['post_failure_count'] += 1
                return False, f"POST failed with status {response.status_code}"
                
        except requests.RequestException as e:
            self.stats['post_failure_count'] += 1
            return False, f"POST request failed: {str(e)}"

    def generate_frames(self):
        """Generate frames for streaming"""
        self.stats['active_connections'] += 1
        
        try:
            while True:
                with self.frame_lock:
                    if self.current_frame is not None:
                        # Encode frame as JPEG
                        ret, buffer = cv2.imencode('.jpg', self.current_frame, 
                                                 [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                            
                            self.stats['frames_streamed'] += 1
                            self.stats['last_update'] = datetime.now().strftime('%H:%M:%S')
                            
                            # Send frame to external server
                            if self.external_server_url:
                                success, message = self.post_to_server(frame=self.current_frame)
                                if not success:
                                    print(f"Failed to POST frame to external server: {message}")
                
                time.sleep(self.frame_interval)
                
        except GeneratorExit:
            self.stats['active_connections'] -= 1
            if self.stats['active_connections'] < 0:
                self.stats['active_connections'] = 0

    def update_frame(self, frame):
        """Update the current frame for streaming"""
        current_time = time.time()
        
        # Limit FPS
        if current_time - self.last_frame_time < self.frame_interval:
            return
        
        self.last_frame_time = current_time
        
        with self.frame_lock:
            self.current_frame = frame.copy()
            if not self.streaming_active:
                self.streaming_active = True
                if not self.stats['start_time']:
                    self.stats['start_time'] = current_time
        
        # Calculate current FPS
        if hasattr(self, '_fps_times'):
            self._fps_times.append(current_time)
            # Keep only last 10 frame times
            self._fps_times = self._fps_times[-10:]
            if len(self._fps_times) > 1:
                fps_duration = self._fps_times[-1] - self._fps_times[0]
                if fps_duration > 0:
                    self.stats['current_fps'] = (len(self._fps_times) - 1) / fps_duration
        else:
            self._fps_times = [current_time]

    def add_overlay_info(self, frame, info_text=None, recognition_results=None):
        """Add overlay information to frame"""
        overlay_frame = frame.copy()
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(overlay_frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add custom info text
        if info_text:
            cv2.putText(overlay_frame, info_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add recognition results
        if recognition_results:
            y_offset = 90
            for user_id, confidence in recognition_results:
                text = f"User: {user_id} ({confidence:.2f})"
                cv2.putText(overlay_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
        
        # Add system status
        status_text = "REGISTRATION MODE" if hasattr(self, '_registration_mode') and self._registration_mode else "MONITORING"
        color = (0, 165, 255) if status_text == "REGISTRATION MODE" else (0, 255, 0)
        cv2.putText(overlay_frame, status_text, (10, overlay_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return overlay_frame

    def set_registration_mode(self, is_registration_mode):
        """Set registration mode status for overlay"""
        self._registration_mode = is_registration_mode

    def start_server(self, debug=False, threaded=True):
        """Start the streaming server"""
        try:
            print(f"Starting live streaming server on http://{self.host}:{self.port}")
            self.app.run(host=self.host, port=self.port, debug=debug, 
                        threaded=threaded, use_reloader=False)
        except Exception as e:
            print(f"Error starting streaming server: {e}")

    def start_server_thread(self):
        """Start streaming server in a separate thread"""
        server_thread = threading.Thread(target=self.start_server, daemon=True)
        server_thread.start()
        print(f"Live streaming server thread started on http://{self.host}:{self.port}")
        return server_thread

    def stop_streaming(self):
        """Stop streaming"""
        self.streaming_active = False
        with self.frame_lock:
            self.current_frame = None
        print("Live streaming stopped")

    def get_stats(self):
        """Get streaming statistics"""
        return self.stats.copy()

    def is_streaming(self):
        """Check if streaming is active"""
        return self.streaming_active

    def get_stream_url(self):
        """Get the streaming URL"""
        return f"http://{self.host}:{self.port}"

    def get_video_feed_url(self):
        """Get the video feed URL"""
        return f"http://{self.host}:{self.port}/video_feed"


# Example usage and integration helper
class StreamingIntegration:
    """Helper class to integrate streaming with face recognition system"""
    
    def __init__(self, face_system, streaming_port=8080, external_server_url=None, external_server_headers=None):
        self.face_system = face_system
        self.streaming = LiveStreaming(port=streaming_port, 
                                     external_server_url=external_server_url,
                                     external_server_headers=external_server_headers)
        self.streaming_thread = None
        
    def start_integrated_streaming(self):
        """Start streaming with face recognition integration"""
        # Start streaming server
        self.streaming_thread = self.streaming.start_server_thread()
        
        # Modify the face system's run method to include streaming
        original_run = self.face_system.run
        
        def run_with_streaming():
            print("Starting face recognition with live streaming...")
            
            try:
                while True:
                    ret, frame = self.face_system.cap.read()
                    if not ret:
                        print("Failed to capture frame")
                        time.sleep(1)
                        continue
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Process face recognition
                    processed_frame, recognized, registration_complete = self.face_system.face_recognition.recognize_face(frame)
                    
                    # Set registration mode status for overlay
                    self.streaming.set_registration_mode(self.face_system.registration_in_progress)
                    
                    # Add overlay information
                    info_text = None
                    if self.face_system.registration_in_progress:
                        info_text = f"Enregistrement en cours: {getattr(self.face_system, 'current_user_info', {}).get('name', 'Utilisateur')}"
                    
                    # Create overlay frame
                    frame_to_display = processed_frame if processed_frame is not None else frame
                    overlay_frame = self.streaming.add_overlay_info(frame_to_display, info_text, recognized)
                    
                    # Update streaming frame
                    self.streaming.update_frame(overlay_frame)
                    
                    # Continue with original face system logic
                    if registration_complete and self.face_system.registration_in_progress:
                        self.face_system.process_registration_completion(True)
                    
                    # Process other face system logic...
                    # (Add the rest of the original run method logic here)
                    
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\nStopping integrated system...")
                self.streaming.stop_streaming()
            finally:
                self.face_system.cleanup()
        
        # Replace the run method
        self.face_system.run = run_with_streaming
        
        return f"Streaming available at: {self.streaming.get_stream_url()}"