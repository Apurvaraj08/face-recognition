#!/usr/bin/env python3
"""
Ultra Enhanced Face Recognition System
Integrates all advanced features: 15+ emotions, advanced detection, tracking, landmarks
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

# Import all advanced modules
from stable_face_recognition import StableFaceRecognition
from alert_system import AlertSystem
from advanced_emotion_detection import AdvancedEmotionDetector
from advanced_face_detection import AdvancedFaceDetector
from face_tracking import AdvancedFaceTracker
from automated_reports import create_automated_report_generator

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class UltraEnhancedSystem:
    def __init__(self):
        print("🚀 Initializing Ultra Enhanced Face Recognition System...")
        
        # Initialize core recognition system
        self.recognition_system = StableFaceRecognition()
        
        # Initialize advanced systems
        self.alert_system = AlertSystem()
        self.advanced_emotion_detector = AdvancedEmotionDetector()
        self.advanced_face_detector = AdvancedFaceDetector()
        self.face_tracker = AdvancedFaceTracker()
        self.report_generator = create_automated_report_generator(
            self.recognition_system, 
            self.alert_system
        )
        
        # Configure systems
        self._configure_systems()
        
        # System state
        self.running = False
        self.current_mode = 'standard'
        
        print("✅ Ultra enhanced system initialized successfully")
    
    def _configure_systems(self):
        """Configure all advanced systems"""
        print("⚙️  Configuring ultra advanced systems...")
        
        # Configure alert system
        self.alert_system.start_alert_system()
        
        # Configure report generator
        self.report_generator.start_scheduler()
        
        print("✅ Ultra systems configured")
    
    def build_reference_database(self):
        """Build reference database"""
        print("📸 Building ultra enhanced reference database...")
        
        if self.recognition_system.build_stable_reference_database():
            print("✅ Reference database built successfully")
            return True
        else:
            print("❌ Failed to build reference database")
            return False
    
    def run_ultra_enhanced_mode(self):
        """Run ultra enhanced face recognition mode with all advanced features"""
        print("\n🎥 Starting Ultra Enhanced Face Recognition Mode")
        print("📝 Controls: 'q' to quit, 's' to save report, 't' to adjust threshold")
        print("🔔 Real-time alerts enabled")
        print("😊 15+ emotion detection enabled")
        print("🎯 Advanced face detection with landmarks")
        print("📊 Face tracking with Kalman filter")
        print("👤 Age & gender estimation")
        print("📐 Face pose estimation")
        
        try:
            self.current_mode = 'ultra_enhanced'
            self.running = True
            
            # Initialize camera
            import cv2
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
                if frame_count % 3 != 0:
                    continue
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                
                # Advanced face detection
                faces = self.advanced_face_detector.detect_faces_advanced(frame)
                
                # Process faces with all advanced features
                current_time = time.time()
                recognitions = []
                
                for (x, y, w, h) in faces:
                    face_region = frame[y:y+h, x:x+w]
                    
                    # Check cooldown
                    face_key = f"{x}_{y}_{w}_{h}"
                    if face_key in recognition_cooldowns:
                        if current_time - recognition_cooldowns[face_key] < 1.5:
                            continue
                    
                    # Advanced quality assessment
                    try:
                        quality_metrics = self.advanced_face_detector.assess_face_quality_advanced(face_region)
                        if isinstance(quality_metrics, dict):
                            overall_quality = quality_metrics.get('overall', 0)
                        else:
                            overall_quality = 0.5  # Default quality
                    except Exception as e:
                        print(f"Error in quality assessment: {e}")
                        overall_quality = 0.5
                    
                    if overall_quality < 0.3:
                        continue  # Skip low quality faces
                    
                    # Facial landmark detection
                    try:
                        landmarks = self.advanced_face_detector.detect_facial_landmarks(face_region)
                    except Exception as e:
                        print(f"Error detecting landmarks: {e}")
                        landmarks = {}
                    
                    # Face pose estimation
                    try:
                        pose = self.advanced_face_detector.estimate_face_pose(face_region, landmarks)
                    except Exception as e:
                        print(f"Error estimating pose: {e}")
                        pose = {'yaw': 0, 'pitch': 0, 'roll': 0}
                    
                    # Age and gender detection
                    try:
                        age_gender = self.advanced_face_detector.detect_age_gender(face_region)
                    except Exception as e:
                        print(f"Error detecting age/gender: {e}")
                        age_gender = {'age_range': 'Unknown', 'gender': 'Unknown', 'confidence': 0.0}
                    
                    # Standard recognition
                    name, similarity = self.recognition_system.recognize_face_stable(face_region)
                    
                    # Advanced emotion detection (15+ emotions)
                    emotion_result = self.advanced_emotion_detector.detect_emotion_from_face(face_region, name)
                    
                    # Create comprehensive result
                    result = {
                        'recognition': {
                            'name': name,
                            'similarity': similarity
                        },
                        'emotion': emotion_result,
                        'quality': quality_metrics,
                        'landmarks': landmarks,
                        'pose': pose,
                        'age_gender': age_gender
                    }
                    
                    recognitions.append(result)
                    
                    # Draw ultra enhanced overlay
                    frame = self._draw_ultra_enhanced_overlay(frame, (x, y, w, h), result)
                    
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
                
                # Update face tracker
                if faces:
                    confirmed_tracks = self.face_tracker.update_with_recognition(faces, recognitions)
                
                # Display stats
                total_recognitions = sum(self.recognition_system.recognition_stats.values())
                tracking_stats = self.face_tracker.get_tracking_statistics()
                
                stats_text = f"Recognitions: {total_recognitions} | Tracks: {tracking_stats.get('confirmed_tracks', 0)} | Threshold: {self.recognition_system.similarity_threshold:.2f}"
                cv2.putText(frame, stats_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Display recognition stats
                y_offset = 60
                for student, count in self.recognition_system.recognition_stats.items():
                    student_text = f"{student}: {count}"
                    cv2.putText(frame, student_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 15
                
                # Display system info
                system_text = "Ultra Enhanced: 15+ Emotions | Landmarks | Tracking | Age/Gender | Pose"
                cv2.putText(frame, system_text, (10, frame.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                cv2.putText(frame, "Press 'q' to quit, 's' to save, 't' to adjust", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.imshow('Ultra Enhanced Face Recognition', frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('s'):
                    self.recognition_system.save_attendance_report()
                    self.advanced_emotion_detector.save_emotion_data()
                    self.advanced_emotion_detector.generate_emotion_report()
                    self.report_generator.generate_custom_report('recognition_summary')
                    print("💾 Comprehensive ultra enhanced report saved!")
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
    
    def _draw_ultra_enhanced_overlay(self, frame, face_location, result):
        """Draw ultra enhanced overlay with all advanced features"""
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
        
        cv2.putText(frame, recognition_text, (x, y - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Emotion info (15+ emotions)
        emotion_result = result['emotion']
        dominant_emotion = emotion_result['dominant_emotion']
        emotion_confidence = emotion_result['confidence']
        
        emotion_colors = {
            'neutral': (128, 128, 128), 'happy': (0, 255, 0), 'sad': (255, 0, 0),
            'angry': (0, 0, 255), 'surprise': (255, 255, 0), 'disgust': (128, 0, 128),
            'fear': (255, 165, 0), 'excited': (0, 255, 255), 'confused': (255, 0, 255),
            'focused': (0, 128, 255), 'tired': (100, 100, 100), 'bored': (150, 150, 150),
            'amused': (0, 200, 100), 'concerned': (200, 100, 0), 'relaxed': (100, 200, 100),
            'frustrated': (200, 50, 50)
        }
        
        emotion_color = emotion_colors.get(dominant_emotion, (255, 255, 255))
        emotion_text = f"{dominant_emotion} ({emotion_confidence:.2f})"
        cv2.putText(frame, emotion_text, (x, y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 2)
        
        # Age and gender
        age_gender = result['age_gender']
        age_text = f"{age_gender['age_range']}"
        gender_text = f"{age_gender['gender']}"
        cv2.putText(frame, age_text, (x, y + h + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, gender_text, (x, y + h + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Pose info
        pose = result['pose']
        pose_text = f"Yaw:{pose['yaw']:.1f} Pitch:{pose['pitch']:.1f}"
        cv2.putText(frame, pose_text, (x, y + h + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Quality score
        quality_data = result.get('quality', {})
        if isinstance(quality_data, dict):
            quality = quality_data.get('overall', 0.5)
        else:
            quality = 0.5
        quality_text = f"Q:{quality:.2f}"
        cv2.putText(frame, quality_text, (x + w - 50, y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw landmarks if available
        landmarks = result.get('landmarks', {})
        if isinstance(landmarks, dict) and landmarks.get('eyes'):
            for eye in landmarks['eyes']:
                if isinstance(eye, dict):
                    ex, ey, ew, eh = eye.get('x', 0), eye.get('y', 0), eye.get('width', 0), eye.get('height', 0)
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 1)
        
        return frame
    
    def run_system_diagnostics(self):
        """Run ultra enhanced system diagnostics"""
        print("\n🔧 Running Ultra Enhanced System Diagnostics...")
        
        diagnostics = {
            'recognition_system': self.recognition_system is not None,
            'alert_system': self.alert_system is not None,
            'advanced_emotion_detector': self.advanced_emotion_detector is not None,
            'advanced_face_detector': self.advanced_face_detector is not None,
            'face_tracker': self.face_tracker is not None,
            'report_generator': self.report_generator is not None
        }
        
        print("\n📊 Ultra Enhanced System Status:")
        for system, status in diagnostics.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {system}: {'Active' if status else 'Inactive'}")
        
        # Test advanced emotion detector
        if self.advanced_emotion_detector:
            print(f"\n😊 Advanced Emotion Detector Stats:")
            print(f"  Emotion Categories: {len(self.advanced_emotion_detector.emotion_labels)}")
            print(f"  Emotions: {', '.join(self.advanced_emotion_detector.emotion_labels)}")
        
        # Test advanced face detector
        if self.advanced_face_detector:
            print(f"\n🎯 Advanced Face Detector Stats:")
            print(f"  Detection Parameters: {self.advanced_face_detector.detection_params}")
            print(f"  Detection Statistics: {self.advanced_face_detector.get_detection_statistics()}")
        
        # Test face tracker
        if self.face_tracker:
            print(f"\n📊 Face Tracker Stats:")
            print(f"  Tracking Statistics: {self.face_tracker.get_tracking_statistics()}")
        
        print("\n✅ Ultra enhanced diagnostics completed")
    
    def cleanup(self):
        """Cleanup and shutdown systems"""
        print("\n🧹 Shutting down ultra enhanced systems...")
        
        try:
            if self.alert_system:
                self.alert_system.stop_alert_system()
            
            if self.report_generator:
                self.report_generator.stop_scheduler()
            
            if self.face_tracker:
                self.face_tracker.reset()
            
            print("✅ Ultra enhanced systems shutdown completed")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Ultra Enhanced Face Recognition System')
    parser.add_argument('--mode', choices=['ultra', 'diagnostics'], 
                       default='ultra', help='Operating mode')
    parser.add_argument('--no-build-db', action='store_true', 
                       help='Skip building reference database')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎥 ULTRA ENHANCED FACE RECOGNITION SYSTEM")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👥 Ultra Advanced Features:")
    print(f"   • 15+ Emotion Detection States")
    print(f"   • Advanced Face Detection with Preprocessing")
    print(f"   • Facial Landmark Detection")
    print(f"   • Face Tracking with Kalman Filter")
    print(f"   • Age & Gender Estimation")
    print(f"   • Face Pose Estimation")
    print(f"   • Advanced Quality Assessment (10 metrics)")
    print(f"📷 Ultra enhanced camera-based detection")
    print("=" * 80)
    
    # Initialize ultra enhanced system
    system = UltraEnhancedSystem()
    
    try:
        # Build reference database
        if not args.no_build_db:
            if not system.build_reference_database():
                print("❌ Failed to start system - reference database not built")
                return
        
        # Run in specified mode
        if args.mode == 'ultra':
            system.run_ultra_enhanced_mode()
        elif args.mode == 'diagnostics':
            system.run_system_diagnostics()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    finally:
        # Cleanup
        system.cleanup()
        print("\n🎉 Ultra enhanced system session completed!")
        print("=" * 80)

if __name__ == "__main__":
    main()
