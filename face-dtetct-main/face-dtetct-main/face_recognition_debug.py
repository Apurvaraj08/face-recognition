"""
Debug version of face recognition - shows detailed matching info
"""

import cv2
import numpy as np
import os
import pickle
from datetime import datetime
from pathlib import Path

# Configuration
STUDENT_IMAGES_DIR = r'C:\Users\Apurava Raj\OneDrive\Grill\Grill Assignment\Student Images'
FEATURES_CACHE = 'data/student_features.pkl'
REPORTS_DIR = 'reports/snapshots'
os.makedirs(REPORTS_DIR, exist_ok=True)

class StudentFaceDatabase:
    """Builds and manages a database of student face features"""
    
    def __init__(self, cache_file=FEATURES_CACHE):
        self.cache_file = cache_file
        self.students = {}
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.orb = cv2.ORB_create(nfeatures=500)
    
    def load_cache(self):
        """Load features from cache"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.students = pickle.load(f)
            print(f'[INFO] Loaded {len(self.students)} students from cache')
            return True
        return False
    
    def match_face_debug(self, face_roi):
        """
        Match a face with detailed debug info.
        Returns (student_name, confidence, all_scores)
        """
        keypoints, descriptors = self.orb.detectAndCompute(face_roi, None)
        
        if descriptors is None or len(descriptors) < 10:
            return None, 0, {}
        
        all_scores = {}
        best_match = None
        best_score = float('inf')
        
        # BFMatcher for ORB (Hamming distance)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        for student_name, data in self.students.items():
            matches_list = []
            
            # Match against all stored descriptors for this student
            for stored_descriptors in data['descriptors_list']:
                try:
                    matches = bf.match(descriptors, stored_descriptors)
                    matches_list.extend(matches)
                except:
                    continue
            
            if not matches_list:
                all_scores[student_name] = float('inf')
                continue
            
            # Calculate metrics
            distances = [m.distance for m in matches_list]
            avg_distance = np.mean(distances)
            min_distance = np.min(distances)
            match_count = len(matches_list)
            
            all_scores[student_name] = {
                'avg_distance': avg_distance,
                'min_distance': min_distance,
                'match_count': match_count,
                'matches': matches_list
            }
            
            if avg_distance < best_score:
                best_score = avg_distance
                best_match = student_name
        
        confidence = max(0, 100 - best_score) if best_match else 0
        
        return best_match, confidence, all_scores


def main():
    """Debug face recognition loop"""
    
    db = StudentFaceDatabase(FEATURES_CACHE)
    
    if not db.load_cache():
        print('[ERROR] Cache not found. Run face_recognition_opencv.py first to build database.')
        return
    
    if not db.students:
        print('[ERROR] No students in database')
        return
    
    # Open camera
    print('[INFO] Opening camera...')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print('[ERROR] Could not open camera')
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f'[SUCCESS] Camera opened. Database has {len(db.students)} students')
    print('[INFO] Press Q to quit, S to save debug info, SPACE to pause')
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30)
            )
        
        # Draw detection info
        cv2.putText(frame, f'Frames processed: {frame_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'Faces detected: {len(faces)}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if paused:
            cv2.putText(frame, 'PAUSED - Press SPACE to resume', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # Match against database
            student_name, confidence, all_scores = db.match_face_debug(face_roi)
            
            # Draw bounding box
            if student_name:
                color = (0, 255, 0)  # Green for recognized
                text = f'{student_name} ({confidence:.0f}%)'
            else:
                color = (0, 0, 255)  # Red for unknown
                text = 'Unknown'
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw all match scores on the side
            y_offset = 120
            cv2.putText(frame, 'Match Scores:', (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            for i, (name, score_info) in enumerate(all_scores.items()):
                if isinstance(score_info, dict):
                    avg_dist = score_info['avg_distance']
                    match_cnt = score_info['match_count']
                    match_text = f'{name}: {avg_dist:.1f} distance ({match_cnt} matches)'
                else:
                    match_text = f'{name}: NO MATCHES'
                
                text_color = (0, 255, 0) if name == student_name else (200, 200, 200)
                cv2.putText(frame, match_text, (20, y_offset + 25 + (i*20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            
            # Draw main result
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow('Face Recognition - DEBUG', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
            state = 'PAUSED' if paused else 'RESUMING'
            print(f'[USER] {state}')
        elif key == ord('s') or key == ord('S'):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = os.path.join(REPORTS_DIR, f'debug_{ts}.jpg')
            cv2.imwrite(fname, frame)
            print(f'[DEBUG] Saved: {fname}')
            print(f'[DEBUG] Frames processed: {frame_count}')
    
    cap.release()
    cv2.destroyAllWindows()
    print(f'[INFO] Closed. Processed {frame_count} frames total.')


if __name__ == '__main__':
    main()
