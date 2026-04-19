"""
Console-based face recognition - outputs results to terminal instead of OpenCV window
"""

import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import sys

# Configuration
STUDENT_IMAGES_DIR = r'C:\Users\Apurava Raj\OneDrive\Grill\Grill Assignment\Student Images'
FEATURES_CACHE = 'data/student_features.pkl'
REPORTS_DIR = 'reports/snapshots'
os.makedirs(REPORTS_DIR, exist_ok=True)

class StudentFaceDatabase:
    def __init__(self, cache_file=FEATURES_CACHE):
        self.cache_file = cache_file
        self.students = {}
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.orb = cv2.ORB_create(nfeatures=500)
    
    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.students = pickle.load(f)
            print(f'[OK] Loaded {len(self.students)} students')
            return True
        print('[ERROR] Cache not found')
        return False
    
    def match_face(self, face_roi):
        keypoints, descriptors = self.orb.detectAndCompute(face_roi, None)
        
        if descriptors is None or len(descriptors) < 10:
            return None, 0
        
        best_match = None
        best_score = float('inf')
        second_best_score = float('inf')
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        for student_name, data in self.students.items():
            matches_list = []
            
            for stored_descriptors in data['descriptors_list']:
                try:
                    matches = bf.knnMatch(descriptors, stored_descriptors, k=2)
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:
                                matches_list.append(m)
                        else:
                            matches_list.append(match_pair[0])
                except:
                    continue
            
            if not matches_list:
                continue
            
            distances = [m.distance for m in matches_list]
            avg_distance = np.mean(distances)
            
            if avg_distance < best_score:
                second_best_score = best_score
                best_score = avg_distance
                best_match = student_name
            elif avg_distance < second_best_score:
                second_best_score = avg_distance
        
        if best_match and (second_best_score == float('inf') or best_score < second_best_score * 0.85):
            confidence = max(0, 100 - best_score)
            return best_match, confidence
        
        return None, 0


def main():
    print('=' * 60)
    print('FACE RECOGNITION SYSTEM - CONSOLE MODE')
    print('=' * 60)
    
    db = StudentFaceDatabase(FEATURES_CACHE)
    if not db.load_cache():
        return
    
    print(f'[OK] Opening camera...')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print('[ERROR] Could not open camera')
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f'[OK] Camera opened')
    print(f'[OK] Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    frame_count = 0
    last_print = 0
    recognized_count = [0, 0, 0]  # [Ansh, Apurva, Brijesh]
    student_names = list(db.students.keys())
    
    print(f'\n[INFO] System ready. Processing frames...')
    print(f'[INFO] Press Ctrl+C to stop\n')
    print(f'[STATUS] Looking for faces...')
    sys.stdout.flush()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('[ERROR] Failed to read frame')
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1
            
            # Detect faces - SENSITIVE params for better detection
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,      # Slower scan = more thorough
                minNeighbors=3,        # Lower = more detections
                minSize=(30, 30),      # Smaller minimum
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Process detected faces
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                student_name, confidence = db.match_face(face_roi)
                
                if student_name:
                    idx = student_names.index(student_name)
                    recognized_count[idx] += 1
            
            # Print statistics every 10 frames
            if frame_count - last_print >= 10:
                status = f"Frames: {frame_count} | Faces detected: {len(faces)} | Recognized: {sum(recognized_count)}"
                breakdown = " | ".join([f"{name}: {count}" for name, count in zip(student_names, recognized_count)])
                print(f"[{status}]")
                print(f"  {breakdown}")
                sys.stdout.flush()
                last_print = frame_count
    
    except KeyboardInterrupt:
        print('\n[INFO] User stopped')
    except Exception as e:
        print(f'[ERROR] {e}')
    
    finally:
        cap.release()
        print(f'\n[SUMMARY] Total frames: {frame_count}')
        print(f'[SUMMARY] Total recognized: {sum(recognized_count)}')
        for name, count in zip(student_names, recognized_count):
            print(f'  - {name}: {count}')
        print('[OK] Closed')


if __name__ == '__main__':
    main()
