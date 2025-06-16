import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from datetime import datetime

class FaceRecognition:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Face detection and mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Face database (local cache)
        self.known_faces = {}
        self.face_db_file = "face_database.pkl"
        self.load_face_database()
        
        # Settings
        self.recognition_threshold = 0.6

    def extract_face_features(self, image, face_landmarks):
        """Extract facial features from MediaPipe landmarks"""
        h, w = image.shape[:2]
        
        # Key facial landmarks indices
        key_points = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        # Extract coordinates
        features = []
        for idx in key_points:
            if idx < len(face_landmarks.landmark):
                x = face_landmarks.landmark[idx].x * w
                y = face_landmarks.landmark[idx].y * h
                features.extend([x, y])
        
        # Normalize features
        if len(features) >= 4:
            face_width = abs(features[0] - features[2])
            face_height = abs(features[1] - features[3])
            
            if face_width > 0 and face_height > 0:
                normalized_features = []
                for i in range(0, len(features), 2):
                    normalized_features.append(features[i] / face_width)
                    normalized_features.append(features[i+1] / face_height)
                return np.array(normalized_features)
        
        return np.array(features)

    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two face feature vectors"""
        if len(features1) != len(features2):
            return 0.0
        
        # Use cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)

    def recognize_face(self, frame):
        """Recognize faces in the frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        recognized_users = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract features
                features = self.extract_face_features(frame, face_landmarks)
                
                if len(features) > 0:
                    # Compare with known faces
                    best_match_id = None
                    best_similarity = 0
                    
                    for user_id, known_features in self.known_faces.items():
                        similarity = self.calculate_similarity(features, known_features)
                        if similarity > best_similarity and similarity > self.recognition_threshold:
                            best_similarity = similarity
                            best_match_id = user_id
                    
                    if best_match_id:
                        recognized_users.append((best_match_id, best_similarity))
                
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)
        
        return frame, recognized_users

    def save_face_database(self):
        """Save face database to file"""
        try:
            with open(self.face_db_file, 'wb') as f:
                pickle.dump(self.known_faces, f)
        except Exception as e:
            print(f"Error saving database: {e}")

    def load_face_database(self):
        """Load face database from file"""
        try:
            if os.path.exists(self.face_db_file):
                with open(self.face_db_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                print(f"Loaded {len(self.known_faces)} faces from local database")
            else:
                print("No existing local face database found")
        except Exception as e:
            print(f"Error loading database: {e}")
            self.known_faces = {}

    def register_face(self, frame, user_id):
        """Register a new face from the frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Use the first detected face
            face_landmarks = results.multi_face_landmarks[0]
            features = self.extract_face_features(frame, face_landmarks)
            
            if len(features) > 0:
                # Store the face features
                self.known_faces[user_id] = features
                self.save_face_database()
                return True
        
        return False