#!/usr/bin/env python3
"""
Super Enhanced Face Recognition System
25+ emotion states with improved face detection and enhanced features
"""

import os
import sys
import time
import threading
import argparse
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import core systems
from stable_face_recognition import StableFaceRecognition
from alert_system import AlertSystem
from expanded_emotion_detection import ExpandedEmotionDetector
from enhanced_face_detection import EnhancedFaceDetector

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class SuperEnhancedSystem:
    def __init__(self):
        print("🚀 Initializing Super Enhanced Face Recognition System...")
        
        # Initialize core systems
        self.recognition_system = StableFaceRecognition()
        self.alert_system = AlertSystem()
        self.expanded_emotion_detector = ExpandedEmotionDetector()
        self.enhanced_face_detector = EnhancedFaceDetector()
        
        # Configure systems
        self._configure_systems()
        
        # System state
        self.running = False
        self.current_mode = 'super_enhanced'
        
        print("✅ Super enhanced system initialized successfully")
    
    def _configure_systems(self):
        """Configure systems"""
        print("⚙️  Configuring super enhanced systems...")
        
        # Configure alert system
        self.alert_system.start_alert_system()
        
        print("✅ Super systems configured")
    
    def build_reference_database(self):
        """Build reference database"""
        print("📸 Building super enhanced reference database...")
        
        if self.recognition_system.build_stable_reference_database():
            print("✅ Reference database built successfully")
            return True
        else:
            print("❌ Failed to build reference database")
            return False
    
    def run_super_enhanced_mode(self):
        """Run super enhanced face recognition mode"""
        print("\n🎥 Starting Super Enhanced Face Recognition Mode")
        print("📝 Controls: 'q' to quit, 's' to save report, 't' to adjust threshold")
        print("🔔 Real-time alerts enabled")
        print("😊 25+ emotion detection states enabled")
        print("🎯 Enhanced face detection with multiple algorithms")
        print("📊 Advanced facial expression analysis")
        print("🔍 Improved preprocessing and quality assessment")
        
        try:
            self.current_mode = 'super_enhanced'
            self.running = True
            
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Could not open camera")
                return False
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            frame_count = 0
            recognition_cooldowns = {}
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % 2 != 0:  # Super enhanced mode processes more frames
                    continue
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                
                # Use enhanced face detection
                faces = self.enhanced_face_detector.detect_faces_enhanced(frame)
                
                # Process faces with super enhanced features
                current_time = time.time()
                recognitions = []
                
                for (x, y, w, h) in faces:
                    face_region = frame[y:y+h, x:x+w]
                    
                    # Check cooldown
                    face_key = f"{x}_{y}_{w}_{h}"
                    if face_key in recognition_cooldowns:
                        if current_time - recognition_cooldowns[face_key] < 1.0:  # Shorter cooldown for better responsiveness
                            continue
                    
                    # Enhanced quality assessment
                    try:
                        quality_metrics = self.enhanced_face_detector.assess_face_quality_enhanced(face_region)
                        if isinstance(quality_metrics, dict):
                            overall_quality = quality_metrics.get('overall', 0)
                        else:
                            overall_quality = 0.5
                    except Exception as e:
                        print(f"Error in quality assessment: {e}")
                        overall_quality = 0.5
                    
                    if overall_quality < 0.2:  # Lower threshold for super enhanced mode
                        continue  # Skip low quality faces
                    
                    # Standard recognition
                    name, similarity = self.recognition_system.recognize_face_stable(face_region)
                    
                    # Super enhanced emotion detection (25+ emotions)
                    try:
                        emotion_result = self.expanded_emotion_detector.detect_emotion_from_face(face_region, name)
                        dominant_emotion = emotion_result['dominant_emotion']
                        emotion_confidence = emotion_result['confidence']
                        top_emotions = emotion_result.get('top_emotions', [])
                    except Exception as e:
                        print(f"Error in emotion detection: {e}")
                        dominant_emotion = "neutral"
                        emotion_confidence = 0.5
                        top_emotions = []
                    
                    # Enhanced facial landmarks
                    try:
                        landmarks = self.enhanced_face_detector.detect_facial_landmarks_enhanced(face_region)
                    except Exception as e:
                        print(f"Error detecting landmarks: {e}")
                        landmarks = {}
                    
                    # Create super enhanced result
                    result = {
                        'recognition': {
                            'name': name,
                            'similarity': similarity
                        },
                        'emotion': {
                            'dominant_emotion': dominant_emotion,
                            'confidence': emotion_confidence,
                            'top_emotions': top_emotions
                        },
                        'quality': quality_metrics,
                        'landmarks': landmarks
                    }
                    
                    recognitions.append(result)
                    
                    # Draw super enhanced overlay
                    frame = self._draw_super_enhanced_overlay(frame, (x, y, w, h), result)
                    
                    # Trigger alerts
                    if name != "Unknown":
                        self.alert_system.trigger_student_alert(name, similarity, 
                                                               self.recognition_system.recognition_stats.get(name, 0))
                        recognition_cooldowns[face_key] = current_time
                    else:
                        self.alert_system.trigger_unknown_alert(similarity)
                    
                    # Update attendance if recognized
                    if name != "Unknown":
                        self.recognition_system.update_attendance(name)
                        self.recognition_system.recognition_stats[name] += 1
                
                # Display stats
                total_recognitions = sum(self.recognition_system.recognition_stats.values())
                detection_stats = self.enhanced_face_detector.get_detection_statistics()
                
                stats_text = f"Recognitions: {total_recognitions} | Faces: {len(faces)} | Threshold: {self.recognition_system.similarity_threshold:.2f}"
                cv2.putText(frame, stats_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Display recognition stats
                y_offset = 60
                for student, count in self.recognition_system.recognition_stats.items():
                    student_text = f"{student}: {count}"
                    cv2.putText(frame, student_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 15
                
                # Display detection stats
                if detection_stats:
                    y_offset += 10
                    for cascade, count in detection_stats.items():
                        if count > 0:
                            cascade_text = f"{cascade}: {count}"
                            cv2.putText(frame, cascade_text, (10, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                            y_offset += 12
                
                # Display system info
                system_text = "Super Enhanced: 25+ Emotions | Multi-Cascade Detection | Expression Analysis"
                cv2.putText(frame, system_text, (10, frame.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                cv2.putText(frame, "Press 'q' to quit, 's' to save, 't' to adjust", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.imshow('Super Enhanced Face Recognition', frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('s'):
                    self.recognition_system.save_attendance_report()
                    self.expanded_emotion_detector.save_emotion_data()
                    self.expanded_emotion_detector.generate_emotion_report()
                    print("💾 Super enhanced report saved!")
                elif key == ord('t'):
                    self.recognition_system.similarity_threshold = (self.recognition_system.similarity_threshold + 0.1) % 1.0
                    print(f"🎯 Threshold adjusted to: {self.recognition_system.similarity_threshold:.2f}")
            
            cap.release()
            cv2.destroyAllWindows()
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.running = False
    
    def _draw_super_enhanced_overlay(self, frame, face_location, result):
        """Draw super enhanced overlay with all advanced features"""
        x, y, w, h = face_location
        
        # Recognition info
        name = result['recognition']['name']
        similarity = result['recognition']['similarity']
        
        if name != "Unknown":
            color = (0, 255, 0) if similarity > 0.7 else (0, 255, 255)
            recognition_text = f"{name} ({similarity:.2f})"
        else:
            color = (0, 0, 255)
            recognition_text = f"Unknown ({similarity:.2f})"
        
        cv2.putText(frame, recognition_text, (x, y - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Emotion info (25+ emotions)
        emotion_result = result['emotion']
        dominant_emotion = emotion_result['dominant_emotion']
        emotion_confidence = emotion_result['confidence']
        top_emotions = emotion_result.get('top_emotions', [])
        
        # Expanded color coding for 25+ emotions
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
        
        emotion_color = emotion_colors.get(dominant_emotion, (255, 255, 255))
        emotion_text = f"{dominant_emotion} ({emotion_confidence:.2f})"
        cv2.putText(frame, emotion_text, (x, y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 2)
        
        # Display top 3 emotions
        if top_emotions and len(top_emotions) > 1:
            y_offset = y - 10
            for i, (emotion, conf) in enumerate(top_emotions[:3]):
                if i > 0:  # Skip dominant (already shown)
                    sub_color = emotion_colors.get(emotion, (200, 200, 200))
                    sub_text = f"{emotion}: {conf:.2f}"
                    cv2.putText(frame, sub_text, (x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, sub_color, 1)
                    y_offset += 12
        
        # Quality score
        quality_data = result.get('quality', {})
        if isinstance(quality_data, dict):
            quality = quality_data.get('overall', 0.5)
        else:
            quality = 0.5
        quality_text = f"Q:{quality:.2f}"
        cv2.putText(frame, quality_text, (x + w - 50, y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw landmarks if available
        landmarks = result.get('landmarks', {})
        if isinstance(landmarks, dict) and landmarks.get('eyes'):
            for eye in landmarks['eyes']:
                if isinstance(eye, dict):
                    ex, ey, ew, eh = eye.get('x', 0), eye.get('y', 0), eye.get('width', 0), eye.get('height', 0)
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 1)
        
        if isinstance(landmarks, dict) and landmarks.get('mouth'):
            for mouth in landmarks['mouth']:
                if isinstance(mouth, dict):
                    mx, my, mw, mh = mouth.get('x', 0), mouth.get('y', 0), mouth.get('width', 0), mouth.get('height', 0)
                    cv2.rectangle(frame, (x + mx, y + my), (x + mx + mw, y + my + mh), (255, 0, 255), 1)
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        return frame
    
    def run_system_diagnostics(self):
        """Run super enhanced system diagnostics"""
        print("\n🔧 Running Super Enhanced System Diagnostics...")
        
        diagnostics = {
            'recognition_system': self.recognition_system is not None,
            'alert_system': self.alert_system is not None,
            'expanded_emotion_detector': self.expanded_emotion_detector is not None,
            'enhanced_face_detector': self.enhanced_face_detector is not None
        }
        
        print("\n📊 Super Enhanced System Status:")
        for system, status in diagnostics.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {system}: {'Active' if status else 'Inactive'}")
        
        # Test expanded emotion detector
        if self.expanded_emotion_detector:
            print(f"\n😊 Expanded Emotion Detector Stats:")
            print(f"  Emotion Categories: {len(self.expanded_emotion_detector.emotion_labels)}")
            print(f"  Emotions: {', '.join(self.expanded_emotion_detector.emotion_labels[:10])}...")
        
        # Test enhanced face detector
        if self.enhanced_face_detector:
            print(f"\n🎯 Enhanced Face Detector Stats:")
            print(f"  Detection Parameters: {self.enhanced_face_detector.detection_params}")
            print(f"  Detection Statistics: {self.enhanced_face_detector.get_detection_statistics()}")
        
        print("\n✅ Super enhanced diagnostics completed")
    
    def cleanup(self):
        """Cleanup and shutdown systems"""
        print("\n🧹 Shutting down super enhanced systems...")
        
        try:
            if self.alert_system:
                self.alert_system.stop_alert_system()
            
            print("✅ Super enhanced systems shutdown completed")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Super Enhanced Face Recognition System')
    parser.add_argument('--mode', choices=['super_enhanced', 'diagnostics'], 
                       default='super_enhanced', help='Operating mode')
    parser.add_argument('--no-build-db', action='store_true', 
                       help='Skip building reference database')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎥 SUPER ENHANCED FACE RECOGNITION SYSTEM")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👥 Super Advanced Features:")
    print(f"   • 25+ Emotion Detection States")
    print(f"   • Enhanced Face Detection (4 cascades)")
    print(f"   • Advanced Facial Expression Analysis")
    print(f"   • Improved Preprocessing & Quality Assessment")
    print(f"   • Real-time Alerts & Landmark Detection")
    print(f"   • Multi-Algorithm Detection")
    print(f"📷 Super enhanced camera-based detection")
    print("=" * 80)
    
    # Initialize super enhanced system
    system = SuperEnhancedSystem()
    
    try:
        # Build reference database
        if not args.no_build_db:
            if not system.build_reference_database():
                print("❌ Failed to start system - reference database not built")
                return
        
        # Run in specified mode
        if args.mode == 'super_enhanced':
            system.run_super_enhanced_mode()
        elif args.mode == 'diagnostics':
            system.run_system_diagnostics()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    finally:
        # Cleanup
        system.cleanup()
        print("\n🎉 Super enhanced system session completed!")
        print("=" * 80)

if __name__ == "__main__":
    main()
