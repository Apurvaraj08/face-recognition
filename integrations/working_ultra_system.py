#!/usr/bin/env python3
"""
Working Ultra Enhanced Face Recognition System
Simplified version with core advanced features working
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
from advanced_emotion_detection import AdvancedEmotionDetector

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class WorkingUltraSystem:
    def __init__(self):
        print("🚀 Initializing Working Ultra Enhanced Face Recognition System...")
        
        # Initialize core systems
        self.recognition_system = StableFaceRecognition()
        self.alert_system = AlertSystem()
        self.emotion_detector = AdvancedEmotionDetector()
        
        # Configure systems
        self._configure_systems()
        
        # System state
        self.running = False
        self.current_mode = 'working_ultra'
        
        print("✅ Working ultra system initialized successfully")
    
    def _configure_systems(self):
        """Configure systems"""
        print("⚙️  Configuring working ultra systems...")
        
        # Configure alert system
        self.alert_system.start_alert_system()
        
        print("✅ Working systems configured")
    
    def build_reference_database(self):
        """Build reference database"""
        print("📸 Building working ultra enhanced reference database...")
        
        if self.recognition_system.build_stable_reference_database():
            print("✅ Reference database built successfully")
            return True
        else:
            print("❌ Failed to build reference database")
            return False
    
    def run_working_ultra_mode(self):
        """Run working ultra enhanced face recognition mode"""
        print("\n🎥 Starting Working Ultra Enhanced Face Recognition Mode")
        print("📝 Controls: 'q' to quit, 's' to save report, 't' to adjust threshold")
        print("🔔 Real-time alerts enabled")
        print("😊 16 emotion detection states enabled")
        print("🎯 Enhanced face detection")
        print("📊 Quality assessment")
        
        try:
            self.current_mode = 'working_ultra'
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
                if frame_count % 3 != 0:
                    continue
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                
                # Use stable face detection
                faces = self.recognition_system.detect_faces_stable(frame)
                
                # Process faces with working advanced features
                current_time = time.time()
                recognitions = []
                
                for (x, y, w, h) in faces:
                    face_region = frame[y:y+h, x:x+w]
                    
                    # Check cooldown
                    face_key = f"{x}_{y}_{w}_{h}"
                    if face_key in recognition_cooldowns:
                        if current_time - recognition_cooldowns[face_key] < 1.5:
                            continue
                    
                    # Basic quality assessment
                    try:
                        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                        sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                        quality = min(1.0, sharpness / 100)
                    except:
                        quality = 0.5
                    
                    if quality < 0.3:
                        continue  # Skip low quality faces
                    
                    # Standard recognition
                    name, similarity = self.recognition_system.recognize_face_stable(face_region)
                    
                    # Advanced emotion detection (16 emotions)
                    try:
                        emotion_result = self.emotion_detector.detect_emotion_from_face(face_region, name)
                        dominant_emotion = emotion_result['dominant_emotion']
                        emotion_confidence = emotion_result['confidence']
                    except Exception as e:
                        print(f"Error in emotion detection: {e}")
                        dominant_emotion = "neutral"
                        emotion_confidence = 0.5
                    
                    # Create working result
                    result = {
                        'recognition': {
                            'name': name,
                            'similarity': similarity
                        },
                        'emotion': {
                            'dominant_emotion': dominant_emotion,
                            'confidence': emotion_confidence
                        },
                        'quality': quality
                    }
                    
                    recognitions.append(result)
                    
                    # Draw working overlay
                    frame = self._draw_working_overlay(frame, (x, y, w, h), result)
                    
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
                
                stats_text = f"Recognitions: {total_recognitions} | Threshold: {self.recognition_system.similarity_threshold:.2f}"
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
                system_text = "Working Ultra: 16 Emotions | Enhanced Detection | Quality Assessment"
                cv2.putText(frame, system_text, (10, frame.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                cv2.putText(frame, "Press 'q' to quit, 's' to save, 't' to adjust", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.imshow('Working Ultra Enhanced Face Recognition', frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('s'):
                    self.recognition_system.save_attendance_report()
                    self.emotion_detector.save_emotion_data()
                    self.emotion_detector.generate_emotion_report()
                    print("💾 Working ultra enhanced report saved!")
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
    
    def _draw_working_overlay(self, frame, face_location, result):
        """Draw working overlay with advanced features"""
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
        
        # Emotion info (16 emotions)
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
        
        # Quality score
        quality = result.get('quality', 0.5)
        quality_text = f"Q:{quality:.2f}"
        cv2.putText(frame, quality_text, (x + w - 50, y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        return frame
    
    def run_system_diagnostics(self):
        """Run working ultra system diagnostics"""
        print("\n🔧 Running Working Ultra Enhanced System Diagnostics...")
        
        diagnostics = {
            'recognition_system': self.recognition_system is not None,
            'alert_system': self.alert_system is not None,
            'emotion_detector': self.emotion_detector is not None
        }
        
        print("\n📊 Working Ultra Enhanced System Status:")
        for system, status in diagnostics.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {system}: {'Active' if status else 'Inactive'}")
        
        # Test emotion detector
        if self.emotion_detector:
            print(f"\n😊 Advanced Emotion Detector Stats:")
            print(f"  Emotion Categories: {len(self.emotion_detector.emotion_labels)}")
            print(f"  Emotions: {', '.join(self.emotion_detector.emotion_labels)}")
        
        print("\n✅ Working ultra enhanced diagnostics completed")
    
    def cleanup(self):
        """Cleanup and shutdown systems"""
        print("\n🧹 Shutting down working ultra enhanced systems...")
        
        try:
            if self.alert_system:
                self.alert_system.stop_alert_system()
            
            print("✅ Working ultra enhanced systems shutdown completed")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Working Ultra Enhanced Face Recognition System')
    parser.add_argument('--mode', choices=['working_ultra', 'diagnostics'], 
                       default='working_ultra', help='Operating mode')
    parser.add_argument('--no-build-db', action='store_true', 
                       help='Skip building reference database')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎥 WORKING ULTRA ENHANCED FACE RECOGNITION SYSTEM")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👥 Working Ultra Advanced Features:")
    print(f"   • 16 Emotion Detection States")
    print(f"   • Enhanced Face Detection")
    print(f"   • Real-time Alerts")
    print(f"   • Quality Assessment")
    print(f"   • Stable Recognition Core")
    print(f"📷 Working ultra enhanced camera-based detection")
    print("=" * 80)
    
    # Initialize working ultra system
    system = WorkingUltraSystem()
    
    try:
        # Build reference database
        if not args.no_build_db:
            if not system.build_reference_database():
                print("❌ Failed to start system - reference database not built")
                return
        
        # Run in specified mode
        if args.mode == 'working_ultra':
            system.run_working_ultra_mode()
        elif args.mode == 'diagnostics':
            system.run_system_diagnostics()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    finally:
        # Cleanup
        system.cleanup()
        print("\n🎉 Working ultra enhanced system session completed!")
        print("=" * 80)

if __name__ == "__main__":
    main()
