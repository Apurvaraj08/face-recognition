#!/usr/bin/env python3
"""
Stable Face Recognition System
Compressed, optimized, and reliable face recognition
"""

import os
import cv2
import numpy as np
import pickle
import json
import csv
import time
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
STUDENT_IMAGES = os.path.join(DATA_DIR, "student_images")
ENCODINGS_FILE = os.path.join(DATA_DIR, "face_encodings.pkl")
REPORTS_DIR = os.path.join(ROOT, "reports")

class StableFaceRecognition:
    def __init__(self):
        # Initialize face detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        
        # Recognition parameters
        self.reference_database = {}
        self.attendance_data = {}
        self.recognition_stats = defaultdict(int)
        self.recognition_cooldown = 1.5
        self.similarity_threshold = 0.6
        
    def extract_stable_features(self, face_img):
        """Extract stable and reliable face features"""
        try:
            # Preprocess face
            face_resized = cv2.resize(face_img, (100, 100))
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            gray_eq = cv2.equalizeHist(gray)
            
            features = {}
            
            # 1. Color histogram features (stable)
            hist_b = cv2.calcHist([face_resized], [0], None, [16], [0, 256])
            hist_g = cv2.calcHist([face_resized], [1], None, [16], [0, 256])
            hist_r = cv2.calcHist([face_resized], [2], None, [16], [0, 256])
            
            hist_b = cv2.normalize(hist_b, hist_b).flatten()
            hist_g = cv2.normalize(hist_g, hist_g).flatten()
            hist_r = cv2.normalize(hist_r, hist_r).flatten()
            
            features['histogram'] = np.concatenate([hist_b, hist_g, hist_r])
            
            # 2. Statistical features (stable)
            features['statistical'] = np.array([
                np.mean(gray_eq), np.std(gray_eq), np.var(gray_eq),
                np.min(gray_eq), np.max(gray_eq), np.median(gray_eq)
            ])
            
            # 3. Simple LBP features (stable)
            height, width = gray_eq.shape
            lbp = np.zeros((height-2, width-2), dtype=np.uint8)
            
            for i in range(1, height-1):
                for j in range(1, width-1):
                    center = gray_eq[i, j]
                    code = 0
                    neighbors = [
                        gray_eq[i-1, j-1], gray_eq[i-1, j], gray_eq[i-1, j+1],
                        gray_eq[i, j+1], gray_eq[i+1, j+1], gray_eq[i+1, j],
                        gray_eq[i+1, j-1], gray_eq[i, j-1]
                    ]
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    lbp[i-1, j-1] = code
            
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            features['lbp'] = lbp_hist
            
            return features
            
        except Exception as e:
            print(f"Error extracting stable features: {e}")
            return None
    
    def calculate_stable_similarity(self, features1, features2):
        """Calculate stable similarity between features"""
        if features1 is None or features2 is None:
            return 0
        
        similarities = []
        
        # Compare each feature type
        for feature_type in ['histogram', 'statistical', 'lbp']:
            if feature_type in features1 and feature_type in features2:
                f1 = features1[feature_type]
                f2 = features2[feature_type]
                
                # Ensure same length
                min_len = min(len(f1), len(f2))
                f1_trimmed = f1[:min_len]
                f2_trimmed = f2[:min_len]
                
                if len(f1_trimmed) > 0:
                    # Correlation similarity
                    if len(f1_trimmed) > 1:
                        correlation = np.corrcoef(f1_trimmed, f2_trimmed)[0, 1]
                        if np.isnan(correlation):
                            correlation = 0
                    else:
                        correlation = 0
                    
                    # Cosine similarity
                    dot_product = np.dot(f1_trimmed, f2_trimmed)
                    norm1 = np.linalg.norm(f1_trimmed)
                    norm2 = np.linalg.norm(f2_trimmed)
                    cosine_sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
                    
                    # Combined similarity
                    feature_similarity = 0.6 * correlation + 0.4 * cosine_sim
                    similarities.append(max(0, min(1, feature_similarity)))
        
        if similarities:
            return np.mean(similarities)
        
        return 0
    
    def detect_faces_stable(self, frame):
        """Stable face detection with multiple cascades"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)
            
            # Use multiple cascades
            faces1 = self.face_cascade.detectMultiScale(gray_eq, 1.1, 5, minSize=(40, 40))
            faces2 = self.face_alt_cascade.detectMultiScale(gray_eq, 1.1, 5, minSize=(40, 40))
            
            # Combine results
            all_faces = list(faces1) + list(faces2)
            
            # Remove duplicates
            unique_faces = []
            for face in all_faces:
                is_duplicate = False
                for unique_face in unique_faces:
                    if (abs(face[0] - unique_face[0]) < 20 and 
                        abs(face[1] - unique_face[1]) < 20 and
                        abs(face[2] - unique_face[2]) < 20 and
                        abs(face[3] - unique_face[3]) < 20):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_faces.append(face)
            
            return unique_faces
            
        except Exception as e:
            print(f"Error in stable face detection: {e}")
            return []
    
    def assess_face_quality_stable(self, face_img):
        """Stable face quality assessment"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Sharpness
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, sharpness / 100)
            
            # Brightness
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # Contrast
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 80)
            
            # Face size
            height, width = gray.shape
            size_score = min(1.0, (height * width) / (80 * 80))
            
            # Combined quality
            quality = (0.3 * sharpness_score + 
                      0.3 * brightness_score + 
                      0.2 * contrast_score + 
                      0.2 * size_score)
            
            return max(0, min(1, quality))
            
        except Exception as e:
            print(f"Error in stable quality assessment: {e}")
            return 0
    
    def build_stable_reference_database(self):
        """Build stable reference database"""
        print("=== Building Stable Reference Database ===")
        
        if not os.path.exists(STUDENT_IMAGES):
            print("❌ Student images directory not found")
            return False
        
        students = ['Ansh', 'Apurva']
        
        for student in students:
            student_path = os.path.join(STUDENT_IMAGES, student)
            if not os.path.exists(student_path):
                continue
            
            images = [f for f in os.listdir(student_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"📸 Processing {student} images...")
            
            student_features = []
            quality_scores = []
            
            for img_file in images:
                img_path = os.path.join(student_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Detect faces
                faces = self.detect_faces_stable(img)
                
                for (x, y, w, h) in faces:
                    face_region = img[y:y+h, x:x+w]
                    quality = self.assess_face_quality_stable(face_region)
                    
                    if quality > 0.3:  # Quality threshold
                        features = self.extract_stable_features(face_region)
                        if features is not None:
                            student_features.append(features)
                            quality_scores.append(quality)
                
                # Save detection visualization
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, f'{student}', (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                result_path = os.path.join(ROOT, f"stable_detection_{student}_{img_file}")
                cv2.imwrite(result_path, img)
            
            if student_features:
                self.reference_database[student] = student_features
                avg_quality = np.mean(quality_scores) if quality_scores else 0
                print(f"✅ {student}: {len(student_features)} feature sets (avg quality: {avg_quality:.2f})")
            else:
                print(f"❌ {student}: No suitable features found")
        
        print(f"📊 Stable database built for {len(self.reference_database)} students")
        return len(self.reference_database) > 0
    
    def recognize_face_stable(self, face_region):
        """Stable face recognition"""
        if not self.reference_database:
            return "Unknown", 0
        
        # Assess quality
        quality = self.assess_face_quality_stable(face_region)
        if quality < 0.3:
            return "Unknown", quality
        
        # Extract features
        features = self.extract_stable_features(face_region)
        if features is None:
            return "Unknown", 0
        
        # Compare with reference database
        best_match = "Unknown"
        best_similarity = 0
        
        for student_name, reference_features_list in self.reference_database.items():
            student_similarities = []
            
            for ref_features in reference_features_list:
                similarity = self.calculate_stable_similarity(features, ref_features)
                student_similarities.append(similarity)
            
            max_similarity = max(student_similarities) if student_similarities else 0
            
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match = student_name
        
        # Apply threshold and quality weighting
        final_similarity = best_similarity * quality
        
        if final_similarity < self.similarity_threshold:
            best_match = "Unknown"
        
        return best_match, final_similarity
    
    def process_camera_feed_stable(self):
        """Process camera feed with stable recognition"""
        print("\n=== Starting Stable Face Recognition ===")
        print("📷 Initializing camera...")
        print("🎯 Press 'q' to quit, 's' to save attendance, 't' to adjust threshold")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        recognition_cooldowns = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Could not read frame")
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % 3 != 0:
                continue
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.detect_faces_stable(frame)
            
            # Process faces
            current_time = time.time()
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                face_region = frame[y:y+h, x:x+w]
                
                # Check cooldown
                face_key = f"{x}_{y}_{w}_{h}"
                if face_key in recognition_cooldowns:
                    if current_time - recognition_cooldowns[face_key] < self.recognition_cooldown:
                        continue
                
                # Recognize face
                name, similarity = self.recognize_face_stable(face_region)
                
                if name != "Unknown":
                    confidence_text = f"{name} ({similarity:.2f})"
                    color = (0, 255, 0) if similarity > 0.7 else (0, 255, 255)
                    
                    self.update_attendance(name)
                    self.recognition_stats[name] += 1
                    recognition_cooldowns[face_key] = current_time
                else:
                    confidence_text = f"Unknown ({similarity:.2f})"
                    color = (0, 0, 255)
                
                cv2.putText(frame, confidence_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)
            
            # Display stats
            total_recognitions = sum(self.recognition_stats.values())
            stats_text = f"Recognitions: {total_recognitions} | Threshold: {self.similarity_threshold:.2f}"
            cv2.putText(frame, stats_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset = 60
            for student, count in self.recognition_stats.items():
                student_text = f"{student}: {count}"
                cv2.putText(frame, student_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            cv2.putText(frame, "Press 'q' to quit, 's' to save, 't' to adjust", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow('Stable Face Recognition', frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_attendance_report()
                print("💾 Attendance report saved!")
            elif key == ord('t'):
                self.similarity_threshold = (self.similarity_threshold + 0.1) % 1.0
                print(f"🎯 Threshold adjusted to: {self.similarity_threshold:.2f}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("📷 Camera feed stopped")
        return True
    
    def update_attendance(self, name):
        """Update attendance data"""
        if name not in self.attendance_data:
            self.attendance_data[name] = {
                'first_seen': datetime.now().strftime('%H:%M:%S'),
                'last_seen': datetime.now().strftime('%H:%M:%S'),
                'detection_count': 0
            }
        
        self.attendance_data[name]['last_seen'] = datetime.now().strftime('%H:%M:%S')
        self.attendance_data[name]['detection_count'] += 1
    
    def save_attendance_report(self):
        """Save attendance report"""
        if not self.attendance_data:
            print("📊 No attendance data to save")
            return
        
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Save CSV
        csv_path = os.path.join(REPORTS_DIR, "stable_attendance_report.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['student', 'first_seen', 'last_seen', 'detection_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for student, data in self.attendance_data.items():
                writer.writerow({
                    'student': student,
                    'first_seen': data['first_seen'],
                    'last_seen': data['last_seen'],
                    'detection_count': data['detection_count']
                })
        
        # Save JSON
        json_path = os.path.join(REPORTS_DIR, "stable_recognition_stats.json")
        stats = {
            'total_recognitions': sum(data['detection_count'] for data in self.attendance_data.values()),
            'unique_students': len(self.attendance_data),
            'similarity_threshold': self.similarity_threshold,
            'recognition_stats': dict(self.recognition_stats),
            'attendance_summary': self.attendance_data
        }
        
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Attendance report saved to: {csv_path}")
        print(f"✅ Statistics saved to: {json_path}")
        
        # Display summary
        print("\n📊 Stable Recognition Summary:")
        print("-" * 50)
        for student, data in self.attendance_data.items():
            print(f"{student:<12} | {data['detection_count']:>3} detections | "
                  f"{data['first_seen']} - {data['last_seen']}")
        print("-" * 50)

def main():
    """Main function for stable face recognition"""
    print("=" * 80)
    print("🎥 STABLE FACE RECOGNITION SYSTEM")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👥 Compressed, optimized, and reliable recognition")
    print(f"📷 Stable camera-based detection")
    print("=" * 80)
    
    # Initialize system
    recognizer = StableFaceRecognition()
    
    # Build reference database
    if not recognizer.build_stable_reference_database():
        print("❌ Failed to build reference database")
        return
    
    print("\n🎯 System ready! Starting stable camera recognition...")
    print("📝 Instructions:")
    print("   • Position face clearly in front of camera")
    print("   • System uses stable algorithms for reliability")
    print("   • Press 's' to save attendance report")
    print("   • Press 't' to adjust recognition threshold")
    print("   • Press 'q' to quit")
    print("\n🚀 Starting stable real-time recognition...")
    
    # Process camera feed
    try:
        recognizer.process_camera_feed_stable()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        recognizer.save_attendance_report()
        print("\n🎉 Stable recognition session completed!")
        print("=" * 80)

if __name__ == "__main__":
    main()
