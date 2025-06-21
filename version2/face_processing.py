import cv2
import numpy as np
import mediapipe as mp

class FaceProcessor:
    def __init__(self):
        # Initialize MediaPipe Face Detection and Face Mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Initialize face detection and mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
    
    def extract_face_landmarks(self, image, face_landmarks):
        """Extract key facial landmarks as feature vector"""
        h, w = image.shape[:2]
        landmarks = []
        
        # Extract key facial points (eyes, nose, mouth corners, etc.)
        key_points = [
            # Left eye
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            # Right eye  
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            # Nose
            1, 2, 5, 4, 6, 168, 8, 9, 10, 151, 195, 197, 196, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 240, 309, 415, 310, 311, 312, 13, 82, 81, 80, 78,
            # Mouth
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318
        ]
        
        for point_idx in key_points:
            if point_idx < len(face_landmarks.landmark):
                x = int(face_landmarks.landmark[point_idx].x * w)
                y = int(face_landmarks.landmark[point_idx].y * h)
                landmarks.extend([x, y])
        
        return np.array(landmarks)
    
    def calculate_face_similarity(self, face1, face2):
        """Calculate similarity between two face encodings"""
        if len(face1) != len(face2):
            return 0
        
        # Normalize the vectors
        face1_norm = face1 / (np.linalg.norm(face1) + 1e-6)
        face2_norm = face2 / (np.linalg.norm(face2) + 1e-6)
        
        # Calculate cosine similarity
        similarity = np.dot(face1_norm, face2_norm)
        return max(0, similarity)  # Ensure non-negative
    
    def process_frame_for_registration(self, frame, rgb_frame):
        """Process frame for face registration"""
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                # Draw bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)
                
                # Get face mesh for this detection
                mesh_results = self.face_mesh.process(rgb_frame)
                if mesh_results.multi_face_landmarks:
                    face_landmarks = mesh_results.multi_face_landmarks[0]
                    return self.extract_face_landmarks(rgb_frame, face_landmarks)
        
        return None
    
    def process_frame_for_recognition(self, frame, rgb_frame, authorized_faces, threshold, consecutive_recognitions, recognition_threshold):
        """Process frame for face recognition"""
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                
                # Get face mesh for recognition
                mesh_results = self.face_mesh.process(rgb_frame)
                if mesh_results.multi_face_landmarks:
                    face_landmarks = mesh_results.multi_face_landmarks[0]
                    face_encoding = self.extract_face_landmarks(rgb_frame, face_landmarks)
                    
                    # Recognize face
                    best_match = None
                    best_similarity = 0
                    
                    for name, auth_encoding in authorized_faces.items():
                        similarity = self.calculate_face_similarity(face_encoding, auth_encoding)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = name
                    
                    if best_similarity > threshold:
                        # Update consecutive recognition counter
                        consecutive_recognitions[best_match] = consecutive_recognitions.get(best_match, 0) + 1
                        
                        # Draw green box for authorized person
                        cv2.rectangle(frame, bbox, (0, 255, 0), 3)
                        
                        # Display recognition info
                        cv2.putText(frame, f"Recognizing: {best_match} ({consecutive_recognitions[best_match]}/{recognition_threshold})", 
                                  (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {best_similarity:.2f}", 
                                  (bbox[0], bbox[1] + bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 1)
                        
                        return best_match, best_similarity, bbox
                    else:
                        # Draw red box for unauthorized person
                        cv2.rectangle(frame, bbox, (0, 0, 255), 3)
                        cv2.putText(frame, "UNAUTHORIZED", 
                                  (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.8, (0, 0, 255), 2)
                        cv2.putText(frame, f"Similarity: {best_similarity:.2f}", 
                                  (bbox[0], bbox[1] + bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 0, 255), 1)
        
        return None
    
    def cleanup(self):
        """Cleanup resources"""
        self.face_detection.close()
        self.face_mesh.close()