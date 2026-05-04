#!/usr/bin/env python3
"""
Emotion Detection System
Adds emotion recognition capabilities to face recognition
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime
from collections import defaultdict

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports")
EMOTION_DATA_DIR = os.path.join(ROOT, "emotion_data")

class EmotionDetector:
    def __init__(self):
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'disgust', 'fear']
        self.emotion_stats = defaultdict(lambda: defaultdict(int))
        self.emotion_history = []
        
        # Emotion detection parameters
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Ensure emotion data directory exists
        os.makedirs(EMOTION_DATA_DIR, exist_ok=True)
        
        # Load emotion models if available
        self.emotion_models = self.load_emotion_models()
    
    def load_emotion_models(self):
        """Load emotion detection models"""
        models = {}
        
        # Try to load pre-trained emotion models
        # For now, we'll use a rule-based approach
        models['rule_based'] = True
        
        return models
    
    def detect_emotions_rule_based(self, face_region):
        """Rule-based emotion detection using facial features"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Extract facial features
            features = self.extract_facial_features(gray)
            
            # Rule-based emotion classification
            emotions = {}
            
            # Neutral (baseline)
            emotions['neutral'] = 0.3
            
            # Happy (mouth curve, eye wrinkles)
            if features.get('mouth_curve', 0) > 0.1:
                emotions['happy'] = 0.6
            else:
                emotions['happy'] = 0.1
            
            # Sad (mouth downturn, eye droop)
            if features.get('mouth_curve', 0) < -0.1:
                emotions['sad'] = 0.5
            else:
                emotions['sad'] = 0.1
            
            # Angry (furrowed brow, tight mouth)
            if features.get('brow_furrow', 0) > 0.2:
                emotions['angry'] = 0.5
            else:
                emotions['angry'] = 0.1
            
            # Surprise (wide eyes, open mouth)
            if features.get('eye_openness', 0) > 0.7:
                emotions['surprise'] = 0.5
            else:
                emotions['surprise'] = 0.1
            
            # Disgust (nose wrinkle, mouth pursed)
            if features.get('nose_wrinkle', 0) > 0.15:
                emotions['disgust'] = 0.4
            else:
                emotions['disgust'] = 0.1
            
            # Fear (wide eyes, tense mouth)
            if features.get('eye_tension', 0) > 0.3:
                emotions['fear'] = 0.4
            else:
                emotions['fear'] = 0.1
            
            # Normalize probabilities
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total for k, v in emotions.items()}
            
            return emotions
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            # Return neutral as fallback
            return {label: (1.0/len(self.emotion_labels)) for label in self.emotion_labels}
    
    def extract_facial_features(self, gray_face):
        """Extract facial features for emotion detection"""
        features = {}
        
        try:
            # Simple feature extraction based on facial regions
            
            height, width = gray_face.shape
            
            # Eye region (top 1/3)
            eye_region = gray_face[:height//3, :]
            
            # Mouth region (bottom 1/3)
            mouth_region = gray_face[2*height//3:, :]
            
            # Nose region (middle 1/3)
            nose_region = gray_face[height//3:2*height//3, :]
            
            # Eye openness (variance in eye region)
            eye_variance = np.var(eye_region)
            features['eye_openness'] = min(1.0, eye_variance / 1000)
            
            # Eye tension (edge density in eye region)
            eye_edges = cv2.Canny(eye_region, 50, 150)
            eye_edge_density = np.sum(eye_edges > 0) / (eye_region.shape[0] * eye_region.shape[1])
            features['eye_tension'] = min(1.0, eye_edge_density * 10)
            
            # Mouth curve (horizontal gradient in mouth region)
            mouth_gradient = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0)
            mouth_curve = np.mean(mouth_gradient) / 255.0
            features['mouth_curve'] = np.clip(mouth_curve, -1, 1)
            
            # Brow furrow (vertical gradient in upper eye region)
            brow_region = gray_face[:height//4, :]
            brow_gradient = cv2.Sobel(brow_region, cv2.CV_64F, 0, 1)
            features['brow_furrow'] = min(1.0, np.mean(np.abs(brow_gradient)) / 100)
            
            # Nose wrinkle (texture in nose region)
            nose_laplacian = cv2.Laplacian(nose_region, cv2.CV_64F)
            features['nose_wrinkle'] = min(1.0, np.var(nose_laplacian) / 1000)
            
        except Exception as e:
            print(f"Error extracting facial features: {e}")
            # Return default values
            features = {
                'eye_openness': 0.5,
                'eye_tension': 0.2,
                'mouth_curve': 0.0,
                'brow_furrow': 0.1,
                'nose_wrinkle': 0.1
            }
        
        return features
    
    def detect_emotion_from_face(self, face_region, student_name="Unknown"):
        """Detect emotion from face region"""
        # Get emotion probabilities
        emotions = self.detect_emotions_rule_based(face_region)
        
        # Get dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        
        # Create emotion result
        emotion_result = {
            'student': student_name,
            'timestamp': datetime.now().isoformat(),
            'emotions': emotions,
            'dominant_emotion': dominant_emotion,
            'confidence': round(confidence, 3)
        }
        
        # Update statistics
        self.emotion_stats[student_name][dominant_emotion] += 1
        self.emotion_history.append(emotion_result)
        
        return emotion_result
    
    def get_emotion_summary(self, student_name=None, time_window_minutes=60):
        """Get emotion summary for a student or all students"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        recent_emotions = [
            emotion for emotion in self.emotion_history
            if datetime.fromisoformat(emotion['timestamp']) > cutoff_time
            and (student_name is None or emotion['student'] == student_name)
        ]
        
        if not recent_emotions:
            return {}
        
        # Count emotions
        emotion_counts = defaultdict(int)
        for emotion in recent_emotions:
            emotion_counts[emotion['dominant_emotion']] += 1
        
        # Calculate percentages
        total_emotions = len(recent_emotions)
        emotion_percentages = {
            emotion: (count / total_emotions) * 100
            for emotion, count in emotion_counts.items()
        }
        
        return {
            'time_window_minutes': time_window_minutes,
            'total_emotions': total_emotions,
            'emotion_counts': dict(emotion_counts),
            'emotion_percentages': emotion_percentages,
            'most_common': max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
        }
    
    def get_student_emotion_profile(self, student_name):
        """Get detailed emotion profile for a student"""
        if student_name not in self.emotion_stats:
            return {}
        
        student_stats = self.emotion_stats[student_name]
        total_detections = sum(student_stats.values())
        
        if total_detections == 0:
            return {}
        
        # Calculate percentages
        emotion_percentages = {
            emotion: (count / total_detections) * 100
            for emotion, count in student_stats.items()
        }
        
        # Get recent summary
        recent_summary = self.get_emotion_summary(student_name, 60)
        
        return {
            'student': student_name,
            'total_detections': total_detections,
            'emotion_percentages': emotion_percentages,
            'most_common_emotion': max(student_stats, key=student_stats.get),
            'recent_summary': recent_summary
        }
    
    def save_emotion_data(self):
        """Save emotion data to file"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        emotion_data = {
            'timestamp': timestamp,
            'emotion_stats': dict(self.emotion_stats),
            'emotion_history': self.emotion_history[-100:],  # Last 100 entries
        }
        
        # Save to file
        emotion_file = os.path.join(EMOTION_DATA_DIR, f"emotion_data_{timestamp}.json")
        with open(emotion_file, 'w') as f:
            json.dump(emotion_data, f, indent=2)
        
        print(f"💾 Emotion data saved to: {emotion_file}")
        return emotion_file
    
    def generate_emotion_report(self, student_name=None):
        """Generate comprehensive emotion report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if student_name:
            # Single student report
            profile = self.get_student_emotion_profile(student_name)
            if not profile:
                return None
            
            report = {
                'report_type': 'student_emotion_profile',
                'timestamp': timestamp,
                'student': student_name,
                'profile': profile
            }
        else:
            # All students report
            student_profiles = {}
            for student in self.emotion_stats:
                profile = self.get_student_emotion_profile(student)
                if profile:
                    student_profiles[student] = profile
            
            # Overall statistics
            all_emotions = defaultdict(int)
            for student_stats in self.emotion_stats.values():
                for emotion, count in student_stats.items():
                    all_emotions[emotion] += count
            
            total_emotions = sum(all_emotions.values())
            overall_percentages = {
                emotion: (count / total_emotions) * 100
                for emotion, count in all_emotions.items()
            } if total_emotions > 0 else {}
            
            report = {
                'report_type': 'comprehensive_emotion_report',
                'timestamp': timestamp,
                'student_profiles': student_profiles,
                'overall_statistics': {
                    'total_emotions': total_emotions,
                    'emotion_percentages': overall_percentages,
                    'most_common_emotion': max(all_emotions, key=all_emotions.get) if all_emotions else None
                }
            }
        
        # Save report
        report_file = os.path.join(REPORTS_DIR, f"emotion_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Emotion report saved to: {report_file}")
        return report_file
    
    def draw_emotion_overlay(self, frame, face_location, emotion_result):
        """Draw emotion detection overlay on frame"""
        x, y, w, h = face_location
        dominant_emotion = emotion_result['dominant_emotion']
        confidence = emotion_result['confidence']
        
        # Color coding for emotions
        emotion_colors = {
            'neutral': (128, 128, 128),
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'surprise': (255, 255, 0),
            'disgust': (128, 0, 128),
            'fear': (255, 165, 0)
        }
        
        color = emotion_colors.get(dominant_emotion, (255, 255, 255))
        
        # Draw emotion label
        emotion_text = f"{dominant_emotion} ({confidence:.2f})"
        cv2.putText(frame, emotion_text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw emotion bar
        bar_width = int(w * confidence)
        cv2.rectangle(frame, (x, y + h + 5), (x + bar_width, y + h + 15), color, -1)
        cv2.rectangle(frame, (x, y + h + 5), (x + w, y + h + 15), color, 2)
        
        return frame

# Integration with main recognition system
class EnhancedRecognitionWithEmotion:
    """Enhanced recognition system with emotion detection"""
    
    def __init__(self, recognition_system):
        self.recognition_system = recognition_system
        self.emotion_detector = EmotionDetector()
    
    def recognize_with_emotion(self, face_region, student_name="Unknown"):
        """Recognize face and detect emotion"""
        # Standard recognition
        name, similarity = self.recognition_system.recognize_face_stable(face_region)
        
        # Emotion detection
        emotion_result = self.emotion_detector.detect_emotion_from_face(face_region, name)
        
        return {
            'recognition': {
                'name': name,
                'similarity': similarity
            },
            'emotion': emotion_result
        }
    
    def process_frame_with_emotion(self, frame):
        """Process frame with both recognition and emotion detection"""
        # Detect faces
        faces = self.recognition_system.detect_faces_stable(frame)
        
        results = []
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            
            # Recognize with emotion
            result = self.recognize_with_emotion(face_region)
            result['face_location'] = (x, y, w, h)
            
            # Update attendance if recognized
            if result['recognition']['name'] != "Unknown":
                self.recognition_system.update_attendance(result['recognition']['name'])
                self.recognition_system.recognition_stats[result['recognition']['name']] += 1
            
            # Draw overlays
            frame = self.draw_enhanced_overlay(frame, result, (x, y, w, h))
            
            results.append(result)
        
        return frame, results
    
    def draw_enhanced_overlay(self, frame, result, face_location):
        """Draw enhanced overlay with recognition and emotion"""
        x, y, w, h = face_location
        
        # Recognition overlay
        name = result['recognition']['name']
        similarity = result['recognition']['similarity']
        
        if name != "Unknown":
            color = (0, 255, 0) if similarity > 0.7 else (0, 255, 255)
            recognition_text = f"{name} ({similarity:.2f})"
        else:
            color = (0, 0, 255)
            recognition_text = f"Unknown ({similarity:.2f})"
        
        cv2.putText(frame, recognition_text, (x, y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Emotion overlay
        emotion_result = result['emotion']
        frame = self.emotion_detector.draw_emotion_overlay(frame, (x, y, w, h), emotion_result)
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        return frame

# Import timedelta for time calculations
from datetime import timedelta
