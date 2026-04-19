"""
Face Recognition using OpenCV Feature Matching (ORB detector)
No dlib required - works with just OpenCV and NumPy

Steps:
1. Loads student images from Student Images folder
2. Extracts ORB features from each student's face
3. When camera detects a face, matches it against stored features
4. Reports which student was identified
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

# Distance threshold for matching (lower = stricter matching)
MATCH_THRESHOLD = 15

class StudentFaceDatabase:
    """Builds and manages a database of student face features"""
    
    def __init__(self, cache_file=FEATURES_CACHE):
        self.cache_file = cache_file
        self.students = {}  # {student_name: {'face': np_array, 'keypoints': [], 'descriptors': []}}
        # Load cascade
        if hasattr(cv2, 'data') and cv2.data.haarcascades:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        else:
            # Fallback for environments where cv2.data is not populated (e.g. some Conda builds)
            import sys
            base_prefix = sys.base_prefix if hasattr(sys, 'base_prefix') else sys.prefix
            cascade_path = os.path.join(base_prefix, 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml')
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.orb = cv2.ORB_create(nfeatures=500)
    
    def build_database(self, student_images_dir):
        """Extract and store face features from student images"""
        print(f'[INFO] Building student face database from {student_images_dir}...')
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for student_folder in os.listdir(student_images_dir):
            student_path = os.path.join(student_images_dir, student_folder)
            
            # Skip non-directory items
            if not os.path.isdir(student_path):
                continue
            
            # Skip special folders
            if student_folder.startswith('.') or student_folder == 'reports':
                continue
            
            student_name = student_folder
            print(f'  [Processing] {student_name}...')
            
            for img_file in os.listdir(student_path):
                if not img_file.lower().endswith(supported_formats):
                    continue
                
                img_path = os.path.join(student_path, img_file)
                
                # Load and detect face
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30)
                )
                
                if len(faces) == 0:
                    continue
                
                # Extract largest face
                (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
                face_roi = gray[y:y+h, x:x+w]
                
                # Extract ORB features
                keypoints, descriptors = self.orb.detectAndCompute(face_roi, None)
                
                if descriptors is None or len(descriptors) < 10:
                    continue
                
                # Store the best features for this student
                if student_name not in self.students:
                    self.students[student_name] = {
                        'descriptors_list': [],
                        'face_count': 0
                    }
                
                self.students[student_name]['descriptors_list'].append(descriptors)
                self.students[student_name]['face_count'] += 1
        
        print(f'[SUCCESS] Loaded {len(self.students)} students:')
        for name, data in self.students.items():
            print(f'  [OK] {name}: {data["face_count"]} image(s)')
        
        self.save_cache()
    
    def save_cache(self):
        """Save features to cache file"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.students, f)
        print(f'[INFO] Saved to {self.cache_file}')
    
    def load_cache(self):
        """Load features from cache"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.students = pickle.load(f)
            print(f'[INFO] Loaded {len(self.students)} students from cache')
            return True
        return False
    
    def match_face(self, face_roi):
        """
        Match a face from camera against student database.
        Returns (student_name, confidence) or (None, 0)
        """
        keypoints, descriptors = self.orb.detectAndCompute(face_roi, None)
        
        if descriptors is None or len(descriptors) < 10:
            return None, 0
        
        best_match = None
        best_score = float('inf')
        second_best_score = float('inf')
        
        # BFMatcher for ORB (Hamming distance)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        for student_name, data in self.students.items():
            matches_list = []
            
            # Match against all stored descriptors for this student
            for stored_descriptors in data['descriptors_list']:
                try:
                    matches = bf.knnMatch(descriptors, stored_descriptors, k=2)
                    # Apply Lowe's ratio test to filter good matches
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:  # Good match threshold
                                matches_list.append(m)
                        else:
                            matches_list.append(match_pair[0])
                except:
                    continue
            
            if not matches_list:
                continue
            
            # Calculate average distance (lower is better)
            distances = [m.distance for m in matches_list]
            avg_distance = np.mean(distances)
            
            # Keep track of second best for ratio test
            if avg_distance < best_score:
                second_best_score = best_score
                best_score = avg_distance
                best_match = student_name
            elif avg_distance < second_best_score:
                second_best_score = avg_distance
        
        # Only report match if it's significantly better than second best
        # This reduces false positives
        if best_match and (second_best_score == float('inf') or best_score < second_best_score * 0.85):
            confidence = max(0, 100 - best_score)
            return best_match, confidence
        
        return None, 0


def main():
    """Main face recognition loop"""
    
    # Load or build database
    db = StudentFaceDatabase(FEATURES_CACHE)
    
    if not db.load_cache():
        print('[INFO] Cache not found. Building from student images...')
        db.build_database(STUDENT_IMAGES_DIR)
    
    if not db.students:
        print('[ERROR] No student faces loaded. Ensure Student Images folder has subdirectories.')
        return
    
    # Open camera with explicit DSHOW backend for Windows
    print('[INFO] Opening camera...')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print('[ERROR] Could not open camera. Trying alternative...')
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('[ERROR] Failed with all methods. Check:')
            print('  1. Camera is physically connected')
            print('  2. No other app is using the camera')
            print('  3. Camera drivers are installed')
            return
    
    # Set camera resolution and buffers
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce lag
    
    print('[SUCCESS] Camera opened')
    print('[INFO] Face recognition ready!')
    print('[INFO] Controls:')
    print('  Q or ESC - Quit')
    print('  S - Save snapshot')
    print('  R - Rebuild database')
    print('[START] Monitoring camera...\n')
    
    # Load face cascade once (not in loop)
    if hasattr(cv2, 'data') and cv2.data.haarcascades:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    else:
        # Fallback for environments where cv2.data is not populated (e.g. some Conda builds)
        import sys
        base_prefix = sys.base_prefix if hasattr(sys, 'base_prefix') else sys.prefix
        cascade_path = os.path.join(base_prefix, 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml')
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    frame_count = 0
    recognition_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('[ERROR] Failed to read frame from camera')
                break
            
            frame = cv2.flip(frame, 1)  # Mirror for natural view
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1
            
            # Detect faces (cascade already loaded)
            # Using stricter parameters to reduce false positives
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.15,      # Slightly larger steps for speed
                minNeighbors=5,        # Stricter: filter out false positives (was 4)
                minSize=(50,50),       # Larger minimum (was 30,30)
                maxSize=(400,400),     # Add maximum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Match against database
                student_name, confidence = db.match_face(face_roi)
                
                # Draw results
                if student_name:
                    color = (0, 255, 0)  # Green for recognized
                    text = f'{student_name} ({confidence:.0f}%)'
                    recognition_count += 1
                else:
                    color = (0, 0, 255)  # Red for unknown
                    text = f'Unknown ({confidence:.0f}%)'
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display info
            cv2.putText(frame, f'Students: {len(db.students)} | Detected: {len(faces)} | Recognized: {recognition_count}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, 'Q=Quit, S=Snapshot, R=Rebuild', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition (OpenCV)', frame)
            
            # Handle keys (with 1ms timeout)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                print('[USER] Quit requested')
                break
            elif key == ord('s') or key == ord('S'):
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                fname = os.path.join(REPORTS_DIR, f'recognized_{ts}.jpg')
                cv2.imwrite(fname, frame)
                print(f'[SNAPSHOT] Saved: {fname}')
            elif key == ord('r') or key == ord('R'):
                print('[USER] Rebuilding database...')
                db.build_database(STUDENT_IMAGES_DIR)
    
    except KeyboardInterrupt:
        print('[USER] Interrupted')
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f'[INFO] Stopped. Processed {frame_count} frames, recognized {recognition_count}.')
        print('[INFO] Face recognition closed.')


if __name__ == '__main__':
    main()
