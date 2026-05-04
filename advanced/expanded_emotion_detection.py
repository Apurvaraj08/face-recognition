#!/usr/bin/env python3
"""
Expanded Emotion Detection System
25+ emotion states with enhanced facial expression analysis
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

class ExpandedEmotionDetector:
    def __init__(self):
        # Expanded emotion categories (25+ emotions)
        self.emotion_labels = [
            # Basic emotions
            'neutral', 'happy', 'sad', 'angry', 'surprise', 'disgust', 'fear',
            
            # Positive emotions
            'excited', 'joyful', 'content', 'proud', 'grateful', 'hopeful', 'confident',
            
            # Negative emotions
            'frustrated', 'worried', 'disappointed', 'lonely', 'ashamed', 'guilty', 'hurt',
            
            # Cognitive emotions
            'confused', 'focused', 'curious', 'thoughtful', 'skeptical',
            
            # Physical states
            'tired', 'bored', 'energetic', 'relaxed', 'stressed'
        ]
        
        self.emotion_stats = defaultdict(lambda: defaultdict(int))
        self.emotion_history = []
        
        # Enhanced emotion detection parameters
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Ensure emotion data directory exists
        os.makedirs(EMOTION_DATA_DIR, exist_ok=True)
        
        # Load emotion models
        self.emotion_models = self.load_emotion_models()
    
    def load_emotion_models(self):
        """Load enhanced emotion detection models"""
        models = {}
        models['rule_based'] = True
        models['facial_features'] = True
        models['expression_analysis'] = True
        return models
    
    def detect_emotions_expanded(self, face_region):
        """Expanded emotion detection with 25+ emotional states"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Extract comprehensive facial features
            features = self.extract_comprehensive_facial_features(gray, face_region)
            
            # Enhanced facial expression analysis
            expression_features = self.analyze_facial_expressions(face_region, gray)
            
            # Combine features
            combined_features = {**features, **expression_features}
            
            # Expanded emotion classification with 25+ emotions
            emotions = self.classify_emotions_expanded(combined_features)
            
            # Normalize probabilities
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total for k, v in emotions.items()}
            
            return emotions
            
        except Exception as e:
            print(f"Error in expanded emotion detection: {e}")
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
            
            # Enhanced eye features
            eye_variance = np.var(eye_region)
            features['eye_openness'] = min(1.0, eye_variance / 1000)
            
            eye_edges = cv2.Canny(eye_region, 50, 150)
            eye_edge_density = np.sum(eye_edges > 0) / (eye_region.shape[0] * eye_region.shape[1])
            features['eye_tension'] = min(1.0, eye_edge_density * 10)
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(eye_region, 1.1, 4)
            features['eye_count'] = len(eyes)
            features['eye_spacing'] = 0.5 if len(eyes) >= 2 else 0.3
            
            # Enhanced mouth features
            mouth_gradient = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0)
            mouth_curve = np.mean(mouth_gradient) / 255.0
            features['mouth_curve'] = np.clip(mouth_curve, -1, 1)
            
            # Detect smile
            smiles = self.mouth_cascade.detectMultiScale(mouth_region, 1.7, 22)
            features['smile_detected'] = 1.0 if len(smiles) > 0 else 0.0
            
            # Enhanced brow features
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
            features = {
                'eye_openness': 0.5, 'eye_tension': 0.2, 'eye_count': 2, 'eye_spacing': 0.5,
                'mouth_curve': 0.0, 'smile_detected': 0.0, 'brow_furrow': 0.1, 'nose_wrinkle': 0.1,
                'forehead_tension': 0.2, 'cheek_flush': 0.3, 'symmetry': 0.5, 'face_energy': 0.4,
                'mouth_openness': 0.3
            }
        
        return features
    
    def analyze_facial_expressions(self, color_face, gray_face):
        """Analyze facial expressions for enhanced emotion detection"""
        expression_features = {}
        
        try:
            height, width = gray_face.shape
            
            # Lip corner detection for smile analysis
            mouth_region = gray_face[2*height//3:, :]
            
            # Detect lip corners using corner detection
            corners = cv2.goodFeaturesToTrack(mouth_region, maxCorners=10, qualityLevel=0.01, minDistance=10)
            if corners is not None:
                corners = np.int0(corners)
                if len(corners) >= 2:
                    # Calculate lip corner curvature
                    left_corner = corners[0][0]
                    right_corner = corners[-1][0]
                    lip_width = right_corner[0] - left_corner[0]
                    lip_center_y = (left_corner[1] + right_corner[1]) / 2
                    
                    # Smile curvature (positive = smile, negative = frown)
                    mid_points = corners[len(corners)//3:2*len(corners)//3]
                    if len(mid_points) > 0:
                        mid_y = np.mean([p[0][1] for p in mid_points])
                        curvature = lip_center_y - mid_y
                        expression_features['lip_curvature'] = min(1.0, max(-1.0, curvature / 20))
                    else:
                        expression_features['lip_curvature'] = 0.0
                else:
                    expression_features['lip_curvature'] = 0.0
            else:
                expression_features['lip_curvature'] = 0.0
            
            # Eye wrinkle detection (for squinting, smiling)
            eye_region = gray_face[:height//3, :]
            eye_edges = cv2.Canny(eye_region, 50, 150)
            crow_lines = np.sum(eye_edges > 0) / (eye_region.shape[0] * eye_region.shape[1])
            expression_features['crow_lines'] = min(1.0, crow_lines * 20)
            
            # Nose bridge analysis (for concentration, anger)
            nose_bridge = gray_face[height//4:height//2, width//2-10:width//2+10]
            if nose_bridge.size > 0:
                nose_intensity = np.mean(nose_bridge)
                expression_features['nose_intensity'] = min(1.0, nose_intensity / 200)
            else:
                expression_features['nose_intensity'] = 0.5
            
            # Jaw tension detection
            jaw_region = gray_face[3*height//4:, :]
            jaw_edges = cv2.Canny(jaw_region, 30, 100)
            jaw_tension = np.sum(jaw_edges > 0) / (jaw_region.shape[0] * jaw_region.shape[1])
            expression_features['jaw_tension'] = min(1.0, jaw_tension * 15)
            
            # Skin color analysis for emotional states
            hsv_face = cv2.cvtColor(color_face, cv2.COLOR_BGR2HSV)
            
            # Red channel intensity (anger, excitement)
            red_intensity = np.mean(hsv_face[:, :, 0])
            expression_features['red_intensity'] = min(1.0, red_intensity / 180)
            
            # Skin brightness (fear, surprise)
            brightness = np.mean(hsv_face[:, :, 2])
            expression_features['skin_brightness'] = min(1.0, brightness / 255)
            
            # Facial muscle tension detection
            face_gradient_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0)
            face_gradient_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1)
            gradient_magnitude = np.sqrt(face_gradient_x**2 + face_gradient_y**2)
            expression_features['muscle_tension'] = min(1.0, np.mean(gradient_magnitude) / 100)
            
        except Exception as e:
            print(f"Error analyzing facial expressions: {e}")
            expression_features = {
                'lip_curvature': 0.0, 'crow_lines': 0.1, 'nose_intensity': 0.5,
                'jaw_tension': 0.2, 'red_intensity': 0.5, 'skin_brightness': 0.5,
                'muscle_tension': 0.3
            }
        
        return expression_features
    
    def classify_emotions_expanded(self, features):
        """Classify emotions with expanded 25+ emotional states"""
        emotions = {}
        
        # Base neutral score
        emotions['neutral'] = 0.15
        
        # BASIC EMOTIONS
        # Happy/Excited/Joyful/Proud/Grateful/Hopeful/Confident
        if features.get('smile_detected', 0) > 0.5:
            smile_intensity = features.get('lip_curvature', 0)
            eye_engagement = features.get('eye_openness', 0.5)
            
            if features.get('face_energy', 0) > 0.7:
                emotions['excited'] = 0.6
                emotions['joyful'] = 0.5
                emotions['happy'] = 0.4
                emotions['proud'] = 0.3
            elif smile_intensity > 0.3 and eye_engagement > 0.6:
                emotions['happy'] = 0.7
                emotions['joyful'] = 0.4
                emotions['grateful'] = 0.3
                emotions['confident'] = 0.3
            else:
                emotions['content'] = 0.5
                emotions['happy'] = 0.4
                emotions['grateful'] = 0.2
        else:
            emotions['happy'] = 0.05
            emotions['excited'] = 0.05
            emotions['joyful'] = 0.05
        
        # Sad/Worried/Disappointed/Lonely/Ashamed/Guilty/Hurt
        if features.get('face_energy', 0) < 0.3 and features.get('smile_detected', 0) < 0.3:
            mouth_curve = features.get('mouth_curve', 0)
            brow_tension = features.get('brow_furrow', 0)
            
            if mouth_curve < -0.2 and brow_tension > 0.3:
                emotions['sad'] = 0.6
                emotions['disappointed'] = 0.4
                emotions['hurt'] = 0.3
            elif features.get('eye_openness', 0) < 0.4:
                emotions['tired'] = 0.5
                emotions['sad'] = 0.3
                emotions['lonely'] = 0.3
            else:
                emotions['worried'] = 0.5
                emotions['sad'] = 0.3
                emotions['guilty'] = 0.2
        else:
            emotions['sad'] = 0.1
            emotions['worried'] = 0.1
            emotions['disappointed'] = 0.05
        
        # Angry/Frustrated/Stressed
        if features.get('brow_furrow', 0) > 0.4 and features.get('eye_tension', 0) > 0.5:
            red_intensity = features.get('red_intensity', 0.5)
            jaw_tension = features.get('jaw_tension', 0.2)
            
            if red_intensity > 0.6 and jaw_tension > 0.4:
                emotions['angry'] = 0.7
                emotions['frustrated'] = 0.4
                emotions['stressed'] = 0.3
            elif features.get('face_energy', 0) > 0.5:
                emotions['frustrated'] = 0.6
                emotions['angry'] = 0.3
                emotions['stressed'] = 0.4
            else:
                emotions['stressed'] = 0.5
                emotions['frustrated'] = 0.3
        else:
            emotions['angry'] = 0.1
            emotions['frustrated'] = 0.1
            emotions['stressed'] = 0.1
        
        # Surprise/Fear
        if features.get('eye_openness', 0) > 0.7 and features.get('mouth_openness', 0) > 0.5:
            skin_brightness = features.get('skin_brightness', 0.5)
            forehead_tension = features.get('forehead_tension', 0)
            
            if skin_brightness > 0.6 and forehead_tension > 0.3:
                emotions['surprise'] = 0.6
                emotions['fear'] = 0.4
            elif forehead_tension > 0.4:
                emotions['fear'] = 0.5
                emotions['surprise'] = 0.3
            else:
                emotions['surprise'] = 0.7
                emotions['hopeful'] = 0.2
        else:
            emotions['surprise'] = 0.1
            emotions['fear'] = 0.1
        
        # Disgust
        if features.get('nose_wrinkle', 0) > 0.3 and features.get('mouth_curve', 0) < -0.1:
            emotions['disgust'] = 0.6
            emotions['disappointed'] = 0.3
        else:
            emotions['disgust'] = 0.1
        
        # COGNITIVE EMOTIONS
        # Confused/Focused/Curious/Thoughtful/Skeptical
        brow_position = features.get('brow_furrow', 0)
        eye_focus = features.get('eye_openness', 0.5)
        muscle_tension = features.get('muscle_tension', 0.3)
        
        if brow_position > 0.2 and eye_focus > 0.5 and muscle_tension > 0.3:
            if features.get('forehead_tension', 0) > 0.4:
                emotions['confused'] = 0.6
                emotions['curious'] = 0.3
            else:
                emotions['focused'] = 0.6
                emotions['thoughtful'] = 0.4
        elif eye_focus > 0.6 and muscle_tension < 0.4:
            emotions['curious'] = 0.5
            emotions['hopeful'] = 0.3
        else:
            emotions['confused'] = 0.1
            emotions['focused'] = 0.1
            emotions['curious'] = 0.1
        
        # Skeptical (combination of confusion and slight negative)
        if brow_position > 0.3 and features.get('mouth_curve', 0) < -0.1:
            emotions['skeptical'] = 0.5
            emotions['confused'] = 0.3
        else:
            emotions['skeptical'] = 0.1
        
        # PHYSICAL STATES
        # Tired/Bored/Energetic/Relaxed
        face_energy = features.get('face_energy', 0.4)
        eye_openness = features.get('eye_openness', 0.5)
        
        if face_energy < 0.2 and eye_openness < 0.4:
            emotions['tired'] = 0.6
            emotions['bored'] = 0.3
        elif face_energy < 0.3 and eye_openness < 0.6:
            emotions['bored'] = 0.5
            emotions['tired'] = 0.2
        elif face_energy > 0.7 and eye_openness > 0.6:
            emotions['energetic'] = 0.6
            emotions['excited'] = 0.3
        elif muscle_tension < 0.3 and brow_position < 0.2:
            emotions['relaxed'] = 0.6
            emotions['content'] = 0.3
        else:
            emotions['tired'] = 0.1
            emotions['bored'] = 0.1
            emotions['energetic'] = 0.1
            emotions['relaxed'] = 0.1
        
        return emotions
    
    def detect_emotion_from_face(self, face_region, student_name="Unknown"):
        """Detect emotion from face region with expanded system"""
        # Get emotion probabilities
        emotions = self.detect_emotions_expanded(face_region)
        
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
    
    def draw_emotion_overlay(self, frame, face_location, emotion_result):
        """Draw expanded emotion detection overlay on frame"""
        x, y, w, h = face_location
        dominant_emotion = emotion_result['dominant_emotion']
        confidence = emotion_result['confidence']
        top_emotions = emotion_result.get('top_emotions', [])
        
        # Expanded color coding for emotions
        emotion_colors = {
            # Basic emotions
            'neutral': (128, 128, 128), 'happy': (0, 255, 0), 'sad': (255, 0, 0),
            'angry': (0, 0, 255), 'surprise': (255, 255, 0), 'disgust': (128, 0, 128),
            'fear': (255, 165, 0),
            
            # Positive emotions
            'excited': (0, 255, 255), 'joyful': (0, 200, 100), 'content': (100, 255, 100),
            'proud': (255, 100, 100), 'grateful': (100, 255, 200), 'hopeful': (100, 200, 255),
            'confident': (0, 150, 255),
            
            # Negative emotions
            'frustrated': (200, 50, 50), 'worried': (150, 100, 0), 'disappointed': (150, 150, 0),
            'lonely': (100, 100, 150), 'ashamed': (150, 0, 75), 'guilty': (100, 50, 50),
            'hurt': (200, 100, 100),
            
            # Cognitive emotions
            'confused': (255, 0, 255), 'focused': (0, 128, 255), 'curious': (255, 128, 0),
            'thoughtful': (128, 128, 255), 'skeptical': (200, 100, 200),
            
            # Physical states
            'tired': (100, 100, 100), 'bored': (150, 150, 150), 'energetic': (255, 200, 0),
            'relaxed': (100, 200, 100), 'stressed': (255, 100, 150)
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
