#!/usr/bin/env python3
"""
Advanced Emotion Detection System
Expanded emotion detection with 15+ emotional states
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime, timedelta
from collections import defaultdict

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports")
EMOTION_DATA_DIR = os.path.join(ROOT, "emotion_data")

class AdvancedEmotionDetector:
    def __init__(self):
        # Expanded emotion categories (15+ emotions)
        self.emotion_labels = [
            'neutral',           # Neutral expression
            'happy',             # Happy/joyful
            'sad',               # Sad/sorrowful
            'angry',             # Angry/furious
            'surprise',          # Surprised/shocked
            'disgust',           # Disgusted/repulsed
            'fear',              # Fearful/scared
            'excited',           # Excited/enthusiastic
            'confused',          # Confused/puzzled
            'focused',           # Focused/concentrating
            'tired',             # Tired/exhausted
            'bored',             # Bored/uninterested
            'amused',            # Amused/entertained
            'concerned',         # Concerned/worried
            'relaxed',           # Relaxed/calm
            'frustrated'         # Frustrated/annoyed
        ]
        
        self.emotion_stats = defaultdict(lambda: defaultdict(int))
        self.emotion_history = []
        
        # Emotion detection parameters
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Ensure emotion data directory exists
        os.makedirs(EMOTION_DATA_DIR, exist_ok=True)
        
        # Load emotion models
        self.emotion_models = self.load_emotion_models()
    
    def load_emotion_models(self):
        """Load emotion detection models"""
        models = {}
        models['rule_based'] = True
        models['facial_features'] = True
        return models
    
    def detect_emotions_advanced(self, face_region):
        """Advanced emotion detection with 15+ emotional states"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Extract comprehensive facial features
            features = self.extract_comprehensive_facial_features(gray, face_region)
            
            # Advanced emotion classification with 15+ emotions
            emotions = self.classify_emotions_advanced(features)
            
            # Normalize probabilities
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total for k, v in emotions.items()}
            
            return emotions
            
        except Exception as e:
            print(f"Error in advanced emotion detection: {e}")
            # Return neutral as fallback
            return {label: (1.0/len(self.emotion_labels)) for label in self.emotion_labels}
    
    def extract_comprehensive_facial_features(self, gray_face, color_face):
        """Extract comprehensive facial features for emotion detection"""
        features = {}
        
        try:
            height, width = gray_face.shape
            
            # Eye region (top 1/3)
            eye_region = gray_face[:height//3, :]
            
            # Mouth region (bottom 1/3)
            mouth_region = gray_face[2*height//3:, :]
            
            # Nose region (middle 1/3)
            nose_region = gray_face[height//3:2*height//3, :]
            
            # Forehead region (top 1/4)
            forehead_region = gray_face[:height//4, :]
            
            # Cheek regions
            left_cheek = gray_face[height//3:2*height//3, :width//2]
            right_cheek = gray_face[height//3:2*height//3, width//2:]
            
            # Eye features
            eye_variance = np.var(eye_region)
            features['eye_openness'] = min(1.0, eye_variance / 1000)
            
            eye_edges = cv2.Canny(eye_region, 50, 150)
            eye_edge_density = np.sum(eye_edges > 0) / (eye_region.shape[0] * eye_region.shape[1])
            features['eye_tension'] = min(1.0, eye_edge_density * 10)
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(eye_region, 1.1, 4)
            features['eye_count'] = len(eyes)
            features['eye_spacing'] = 0.5 if len(eyes) >= 2 else 0.3
            
            # Mouth features
            mouth_gradient = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0)
            mouth_curve = np.mean(mouth_gradient) / 255.0
            features['mouth_curve'] = np.clip(mouth_curve, -1, 1)
            
            # Detect smile
            smiles = self.mouth_cascade.detectMultiScale(mouth_region, 1.7, 22)
            features['smile_detected'] = 1.0 if len(smiles) > 0 else 0.0
            
            # Brow features
            brow_region = gray_face[:height//4, :]
            brow_gradient = cv2.Sobel(brow_region, cv2.CV_64F, 0, 1)
            features['brow_furrow'] = min(1.0, np.mean(np.abs(brow_gradient)) / 100)
            
            # Nose features
            nose_laplacian = cv2.Laplacian(nose_region, cv2.CV_64F)
            features['nose_wrinkle'] = min(1.0, np.var(nose_laplacian) / 1000)
            
            # Forehead features (for worry/concern)
            forehead_edges = cv2.Canny(forehead_region, 30, 100)
            forehead_tension = np.sum(forehead_edges > 0) / (forehead_region.shape[0] * forehead_region.shape[1])
            features['forehead_tension'] = min(1.0, forehead_tension * 15)
            
            # Cheek features (for blushing/flushing)
            left_cheek_color = np.mean(color_face[height//3:2*height//3, :width//2, 1])  # Green channel
            right_cheek_color = np.mean(color_face[height//3:2*height//3, width//2:, 1])
            features['cheek_flush'] = min(1.0, (left_cheek_color + right_cheek_color) / 200)
            
            # Overall face symmetry
            left_half = gray_face[:, :width//2]
            right_half = np.fliplr(gray_face[:, width//2:])
            if left_half.shape == right_half.shape:
                correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
                features['symmetry'] = max(0, correlation) if not np.isnan(correlation) else 0.5
            else:
                features['symmetry'] = 0.5
            
            # Face energy (overall activity)
            face_laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
            features['face_energy'] = min(1.0, np.var(face_laplacian) / 500)
            
            # Mouth openness
            mouth_height = mouth_region.shape[0]
            mouth_width = mouth_region.shape[1]
            mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
            features['mouth_openness'] = min(1.0, mouth_aspect_ratio * 2)
            
        except Exception as e:
            print(f"Error extracting comprehensive facial features: {e}")
            # Return default values
            features = {
                'eye_openness': 0.5,
                'eye_tension': 0.2,
                'eye_count': 2,
                'eye_spacing': 0.5,
                'mouth_curve': 0.0,
                'smile_detected': 0.0,
                'brow_furrow': 0.1,
                'nose_wrinkle': 0.1,
                'forehead_tension': 0.2,
                'cheek_flush': 0.3,
                'symmetry': 0.5,
                'face_energy': 0.4,
                'mouth_openness': 0.3
            }
        
        return features
    
    def classify_emotions_advanced(self, features):
        """Classify emotions based on comprehensive features"""
        emotions = {}
        
        # Base neutral score
        emotions['neutral'] = 0.2
        
        # Happy/Excited/Amused (high smile, high energy, low tension)
        if features.get('smile_detected', 0) > 0.5:
            if features.get('face_energy', 0) > 0.6:
                emotions['excited'] = 0.7
                emotions['happy'] = 0.5
                emotions['amused'] = 0.4
            else:
                emotions['happy'] = 0.7
                emotions['amused'] = 0.5
                emotions['excited'] = 0.3
        else:
            emotions['happy'] = 0.1
            emotions['excited'] = 0.1
            emotions['amused'] = 0.1
        
        # Sad/Tired/Bored (low energy, low smile, droopy features)
        if features.get('face_energy', 0) < 0.3 and features.get('smile_detected', 0) < 0.3:
            if features.get('eye_openness', 0) < 0.4:
                emotions['tired'] = 0.6
                emotions['sad'] = 0.4
                emotions['bored'] = 0.3
            else:
                emotions['sad'] = 0.5
                emotions['bored'] = 0.4
                emotions['tired'] = 0.2
        else:
            emotions['sad'] = 0.1
            emotions['tired'] = 0.1
            emotions['bored'] = 0.1
        
        # Angry/Frustrated (high brow furrow, high tension, low smile)
        if features.get('brow_furrow', 0) > 0.3 and features.get('eye_tension', 0) > 0.4:
            if features.get('face_energy', 0) > 0.5:
                emotions['angry'] = 0.7
                emotions['frustrated'] = 0.5
            else:
                emotions['frustrated'] = 0.6
                emotions['angry'] = 0.4
        else:
            emotions['angry'] = 0.1
            emotions['frustrated'] = 0.1
        
        # Surprise/Shock (wide eyes, open mouth, high energy)
        if features.get('eye_openness', 0) > 0.7 and features.get('mouth_openness', 0) > 0.5:
            emotions['surprise'] = 0.7
            emotions['excited'] = 0.3
        else:
            emotions['surprise'] = 0.1
        
        # Fear/Concerned (high forehead tension, wide eyes, low smile)
        if features.get('forehead_tension', 0) > 0.5 and features.get('eye_openness', 0) > 0.5:
            if features.get('face_energy', 0) > 0.4:
                emotions['fear'] = 0.6
                emotions['concerned'] = 0.4
            else:
                emotions['concerned'] = 0.6
                emotions['fear'] = 0.3
        else:
            emotions['fear'] = 0.1
            emotions['concerned'] = 0.1
        
        # Disgust (nose wrinkle, mouth downturn, cheek flush)
        if features.get('nose_wrinkle', 0) > 0.2 and features.get('mouth_curve', 0) < -0.1:
            emotions['disgust'] = 0.6
        else:
            emotions['disgust'] = 0.1
        
        # Confused (asymmetric features, moderate tension)
        if features.get('symmetry', 0.5) < 0.3 and features.get('brow_furrow', 0) > 0.2:
            emotions['confused'] = 0.5
        else:
            emotions['confused'] = 0.1
        
        # Focused (moderate energy, low smile, eyes present)
        if (features.get('face_energy', 0) > 0.3 and 
            features.get('smile_detected', 0) < 0.3 and 
            features.get('eye_count', 0) >= 2):
            emotions['focused'] = 0.6
        else:
            emotions['focused'] = 0.1
        
        # Relaxed (low tension, moderate symmetry, neutral expression)
        if (features.get('eye_tension', 0) < 0.3 and 
            features.get('brow_furrow', 0) < 0.2 and 
            features.get('symmetry', 0.5) > 0.6):
            emotions['relaxed'] = 0.6
        else:
            emotions['relaxed'] = 0.1
        
        return emotions
    
    def detect_emotion_from_face(self, face_region, student_name="Unknown"):
        """Detect emotion from face region with advanced system"""
        # Get emotion probabilities
        emotions = self.detect_emotions_advanced(face_region)
        
        # Get dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        
        # Get top 3 emotions
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotions = sorted_emotions[:3]
        
        # Create emotion result
        emotion_result = {
            'student': student_name,
            'timestamp': datetime.now().isoformat(),
            'emotions': emotions,
            'dominant_emotion': dominant_emotion,
            'confidence': round(confidence, 3),
            'top_emotions': [(e, round(c, 3)) for e, c in top_emotions],
            'emotion_count': len(self.emotion_labels)
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
            'most_common': max(emotion_counts, key=emotion_counts.get) if emotion_counts else None,
            'emotion_diversity': len(emotion_counts)
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
        
        # Emotion patterns
        emotion_patterns = self._analyze_emotion_patterns(student_name)
        
        return {
            'student': student_name,
            'total_detections': total_detections,
            'emotion_percentages': emotion_percentages,
            'most_common_emotion': max(student_stats, key=student_stats.get),
            'recent_summary': recent_summary,
            'emotion_patterns': emotion_patterns,
            'emotional_range': len(student_stats)
        }
    
    def _analyze_emotion_patterns(self, student_name):
        """Analyze emotion patterns for a student"""
        student_emotions = [
            emotion for emotion in self.emotion_history
            if emotion['student'] == student_name
        ]
        
        if len(student_emotions) < 3:
            return {}
        
        # Analyze transitions
        transitions = defaultdict(int)
        for i in range(len(student_emotions) - 1):
            from_emotion = student_emotions[i]['dominant_emotion']
            to_emotion = student_emotions[i + 1]['dominant_emotion']
            transitions[f"{from_emotion} -> {to_emotion}"] += 1
        
        # Most common transitions
        common_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_transitions': len(student_emotions) - 1,
            'common_transitions': dict(common_transitions),
            'emotion_stability': 1.0 - (len(transitions) / (len(student_emotions) - 1)) if len(student_emotions) > 1 else 1.0
        }
    
    def save_emotion_data(self):
        """Save emotion data to file"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        emotion_data = {
            'timestamp': timestamp,
            'emotion_stats': dict(self.emotion_stats),
            'emotion_history': self.emotion_history[-100:],  # Last 100 entries
            'emotion_labels': self.emotion_labels,
            'total_emotions': len(self.emotion_labels)
        }
        
        # Save to file
        emotion_file = os.path.join(EMOTION_DATA_DIR, f"advanced_emotion_data_{timestamp}.json")
        with open(emotion_file, 'w') as f:
            json.dump(emotion_data, f, indent=2)
        
        print(f"💾 Advanced emotion data saved to: {emotion_file}")
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
                'profile': profile,
                'emotion_categories': len(self.emotion_labels)
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
                    all_emotions[emotion] += 1
            
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
                    'most_common_emotion': max(all_emotions, key=all_emotions.get) if all_emotions else None,
                    'emotion_diversity': len(all_emotions)
                },
                'emotion_categories': len(self.emotion_labels)
            }
        
        # Save report
        report_file = os.path.join(REPORTS_DIR, f"advanced_emotion_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Advanced emotion report saved to: {report_file}")
        return report_file
    
    def draw_emotion_overlay(self, frame, face_location, emotion_result):
        """Draw advanced emotion detection overlay on frame"""
        x, y, w, h = face_location
        dominant_emotion = emotion_result['dominant_emotion']
        confidence = emotion_result['confidence']
        top_emotions = emotion_result.get('top_emotions', [])
        
        # Color coding for emotions (expanded palette)
        emotion_colors = {
            'neutral': (128, 128, 128),
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'surprise': (255, 255, 0),
            'disgust': (128, 0, 128),
            'fear': (255, 165, 0),
            'excited': (0, 255, 255),
            'confused': (255, 0, 255),
            'focused': (0, 128, 255),
            'tired': (100, 100, 100),
            'bored': (150, 150, 150),
            'amused': (0, 200, 100),
            'concerned': (200, 100, 0),
            'relaxed': (100, 200, 100),
            'frustrated': (200, 50, 50)
        }
        
        color = emotion_colors.get(dominant_emotion, (255, 255, 255))
        
        # Draw dominant emotion label
        emotion_text = f"{dominant_emotion} ({confidence:.2f})"
        cv2.putText(frame, emotion_text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw top 3 emotions if available
        if top_emotions and len(top_emotions) > 1:
            y_offset = y + h + 20
            for i, (emotion, conf) in enumerate(top_emotions[:3]):
                if i > 0:  # Skip dominant (already shown)
                    sub_text = f"{emotion}: {conf:.2f}"
                    cv2.putText(frame, sub_text, (x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset += 15
        
        # Draw emotion confidence bar
        bar_width = int(w * confidence)
        cv2.rectangle(frame, (x, y + h + 5), (x + bar_width, y + h + 10), color, -1)
        cv2.rectangle(frame, (x, y + h + 5), (x + w, y + h + 10), color, 2)
        
        return frame
