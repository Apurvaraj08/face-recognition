#!/usr/bin/env python3
"""
Enhanced Main Face Recognition System
Integrates all advanced functionalities
"""

import os
import sys
import time
import threading
import argparse
from datetime import datetime
from collections import defaultdict

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import all advanced modules
from stable_face_recognition import StableFaceRecognition
from alert_system import AlertSystem
from analytics_dashboard import AnalyticsDashboard
from multi_camera_system import MultiCameraSystem
from face_recognition_api import FaceRecognitionAPI
from emotion_detection import EmotionDetector, EnhancedRecognitionWithEmotion
from batch_processor import BatchProcessor, BatchScheduler
from automated_reports import create_automated_report_generator

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class EnhancedMainSystem:
    def __init__(self):
        print("🚀 Initializing Enhanced Face Recognition System...")
        
        # Initialize core recognition system
        self.recognition_system = StableFaceRecognition()
        
        # Initialize advanced systems
        self.alert_system = AlertSystem()
        self.emotion_detector = EmotionDetector()
        self.enhanced_recognition = EnhancedRecognitionWithEmotion(self.recognition_system)
        self.batch_processor = BatchProcessor(self.recognition_system)
        self.report_generator = create_automated_report_generator(
            self.recognition_system, 
            self.alert_system
        )
        self.multi_camera_system = MultiCameraSystem()
        self.api_server = FaceRecognitionAPI()
        
        # Configure systems
        self._configure_systems()
        
        # System state
        self.running = False
        self.current_mode = 'standard'
        
        print("✅ Enhanced system initialized successfully")
    
    def _configure_systems(self):
        """Configure all advanced systems"""
        print("⚙️  Configuring advanced systems...")
        
        # Configure alert system
        self.alert_system.start_alert_system()
        
        # Configure report generator
        self.report_generator.start_scheduler()
        
        # Configure API server (but don't start it yet)
        self.api_server.initialize_systems(
            self.recognition_system,
            self.alert_system,
            None  # Analytics system would be integrated here
        )
        
        print("✅ Systems configured")
    
    def build_reference_database(self):
        """Build reference database"""
        print("📸 Building enhanced reference database...")
        
        if self.recognition_system.build_stable_reference_database():
            print("✅ Reference database built successfully")
            return True
        else:
            print("❌ Failed to build reference database")
            return False
    
    def run_standard_mode(self):
        """Run standard face recognition mode"""
        print("\n🎥 Starting Standard Face Recognition Mode")
        print("📝 Controls: 'q' to quit, 's' to save report, 't' to adjust threshold")
        print("🔔 Real-time alerts enabled")
        print("😊 Emotion detection enabled")
        
        try:
            self.current_mode = 'standard'
            self.running = True
            
            while self.running:
                # Initialize camera
                import cv2
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("❌ Could not open camera")
                    break
                
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
                    
                    # Detect faces
                    faces = self.recognition_system.detect_faces_stable(frame)
                    
                    # Process faces with enhanced recognition
                    current_time = time.time()
                    for (x, y, w, h) in faces:
                        face_region = frame[y:y+h, x:x+w]
                        
                        # Check cooldown
                        face_key = f"{x}_{y}_{w}_{h}"
                        if face_key in recognition_cooldowns:
                            if current_time - recognition_cooldowns[face_key] < 1.5:
                                continue
                        
                        # Enhanced recognition with emotion
                        result = self.enhanced_recognition.recognize_with_emotion(face_region)
                        name = result['recognition']['name']
                        similarity = result['recognition']['similarity']
                        emotion_result = result['emotion']
                        
                        # Draw enhanced overlay
                        frame = self.enhanced_recognition.draw_enhanced_overlay(frame, result, (x, y, w, h))
                        
                        # Trigger alerts
                        if name != "Unknown":
                            self.alert_system.trigger_student_alert(name, similarity, 
                                                           self.recognition_system.recognition_stats.get(name, 0))
                            recognition_cooldowns[face_key] = current_time
                        else:
                            self.alert_system.trigger_unknown_alert(similarity)
                    
                        # Update analytics
                        if hasattr(self, 'analytics_system') and self.analytics_system:
                            self.analytics_system.add_recognition_event(name, similarity, 
                                                               emotion_result.get('quality', 0))
                    
                    # Display stats
                    total_recognitions = sum(self.recognition_system.recognition_stats.values())
                    stats_text = f"Recognitions: {total_recognitions} | Threshold: {self.recognition_system.similarity_threshold:.2f}"
                    cv2.putText(frame, stats_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Display recognition stats
                    y_offset = 60
                    for student, count in self.recognition_system.recognition_stats.items():
                        student_text = f"{student}: {count}"
                        cv2.putText(frame, student_text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += 20
                    
                    cv2.putText(frame, "Press 'q' to quit, 's' to save, 't' to adjust", (10, frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    cv2.imshow('Enhanced Face Recognition', frame)
                    
                    # Handle input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                        break
                    elif key == ord('s'):
                        self.recognition_system.save_attendance_report()
                        self.emotion_detector.save_emotion_data()
                        self.report_generator.generate_custom_report('recognition_summary')
                        print("💾 Comprehensive report saved!")
                    elif key == ord('t'):
                        self.recognition_system.similarity_threshold = (self.recognition_system.similarity_threshold + 0.1) % 1.0
                        print(f"🎯 Threshold adjusted to: {self.recognition_system.similarity_threshold:.2f}")
                    elif key == ord('a'):
                        # Show analytics dashboard
                        self._show_analytics_dashboard()
                    elif key == ord('b'):
                        # Batch processing mode
                        self._run_batch_processing_mode()
                
                cap.release()
                cv2.destroyAllWindows()
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.running = False
    
    def run_multi_camera_mode(self):
        """Run multi-camera mode"""
        print("\n🎥 Starting Multi-Camera Mode")
        
        try:
            self.current_mode = 'multi_camera'
            self.multi_camera_system.run_multi_camera_system(self.recognition_system)
        except Exception as e:
            print(f"❌ Error in multi-camera mode: {e}")
    
    def run_api_mode(self):
        """Run API server mode"""
        print("\n🌐 Starting API Server Mode")
        print("📋 API endpoints available at http://localhost:5000/api")
        
        try:
            self.current_mode = 'api'
            self.api_server.run(host='0.0.0.0', port=5000, debug=False)
        except Exception as e:
            print(f"❌ Error in API mode: {e}")
    
    def run_batch_mode(self):
        """Run batch processing mode"""
        print("\n📦 Starting Batch Processing Mode")
        
        try:
            self.current_mode = 'batch'
            self._run_batch_processing_mode()
        except Exception as e:
            print(f"❌ Error in batch mode: {e}")
    
    def _run_batch_processing_mode(self):
        """Run batch processing"""
        from batch_processor import run_batch_processing_example
        
        # Process student images directory
        student_images_dir = os.path.join(ROOT, "data", "student_images")
        if os.path.exists(student_images_dir):
            print(f"📁 Processing directory: {student_images_dir}")
            results = self.batch_processor.process_directory(student_images_dir, recursive=False)
            
            if results:
                print("✅ Batch processing completed")
                self.batch_processor.generate_batch_report()
            else:
                print("❌ No results from batch processing")
        else:
            print("❌ Student images directory not found")
        
        # Show example usage
        run_batch_processing_example()
    
    def _show_analytics_dashboard(self):
        """Show analytics dashboard"""
        try:
            from analytics_dashboard import run_dashboard
            print("📊 Opening Analytics Dashboard...")
            run_dashboard()
        except Exception as e:
            print(f"❌ Error opening analytics dashboard: {e}")
    
    def run_system_diagnostics(self):
        """Run system diagnostics"""
        print("\n🔧 Running System Diagnostics...")
        
        diagnostics = {
            'recognition_system': self.recognition_system is not None,
            'alert_system': self.alert_system is not None,
            'emotion_detector': self.emotion_detector is not None,
            'batch_processor': self.batch_processor is not None,
            'report_generator': self.report_generator is not None,
            'multi_camera_system': self.multi_camera_system is not None,
            'api_server': self.api_server is not None
        }
        
        print("\n📊 System Status:")
        for system, status in diagnostics.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {system}: {'Active' if status else 'Inactive'}")
        
        # Test recognition system
        if self.recognition_system:
            print(f"\n📈 Recognition System Stats:")
            print(f"  Reference Database: {len(self.recognition_system.reference_database)} students")
            print(f"  Recognition Stats: {dict(self.recognition_system.recognition_stats)}")
            print(f"  Attendance Data: {len(self.recognition_system.attendance_data)} students")
        
        # Test alert system
        if self.alert_system:
            print(f"\n🔔 Alert System Stats:")
            alert_summary = self.alert_system.get_alert_summary()
            print(f"  Total Alerts: {alert_summary.get('total_alerts', 0)}")
            print(f"  Alert Types: {list(alert_summary.get('alert_types', {}).keys())}")
        
        print("\n✅ Diagnostics completed")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive system report"""
        print("\n📊 Generating Comprehensive Report...")
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M-%S')
        
        report = {
            'timestamp': timestamp,
            'system_mode': self.current_mode,
            'recognition_system': {
                'reference_database_size': len(self.recognition_system.reference_database),
                'recognition_stats': dict(self.recognition_system.recognition_stats),
                'attendance_data': dict(self.recognition_system.attendance_data)
            },
            'alert_system': self.alert_system.get_alert_summary(),
            'emotion_detection': {
                'emotion_stats': dict(self.emotion_detector.emotion_stats),
                'total_emotions': len(self.emotion_detector.emotion_history)
            },
            'batch_processing': {
                'processing_stats': dict(self.batch_processor.processing_stats)
            }
        }
        
        # Save comprehensive report
        import json
        report_path = os.path.join(ROOT, f"comprehensive_report_{timestamp.replace(':', '-')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Comprehensive report saved to: {report_path}")
        
        # Display summary
        print(f"\n📊 System Report Summary ({timestamp}):")
        print("-" * 50)
        print(f"Mode: {self.current_mode}")
        print(f"Students in Database: {len(self.recognition_system.reference_database)}")
        print(f"Total Recognitions: {sum(self.recognition_system.recognition_stats.values())}")
        print(f"Active Students: {len(self.recognition_system.attendance_data)}")
        print(f"Total Alerts: {self.alert_system.get_alert_summary().get('total_alerts', 0)}")
        print(f"Emotion Records: {len(self.emotion_detector.emotion_history)}")
        print("-" * 50)
        
        return report_path
    
    def cleanup(self):
        """Cleanup and shutdown systems"""
        print("\n🧹 Shutting down enhanced systems...")
        
        try:
            if self.alert_system:
                self.alert_system.stop_alert_system()
            
            if self.report_generator:
                self.report_generator.stop_scheduler()
            
            print("✅ Systems shutdown completed")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Face Recognition System')
    parser.add_argument('--mode', choices=['standard', 'multi', 'api', 'batch', 'diagnostics'], 
                       default='standard', help='Operating mode')
    parser.add_argument('--no-build-db', action='store_true', 
                       help='Skip building reference database')
    parser.add_argument('--report', action='store_true', 
                       help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎥 ENHANCED FACE RECOGNITION SYSTEM")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👥 Advanced features: Alerts, Analytics, Multi-Camera, API, Emotion Detection")
    print(f"📷 Enhanced camera-based detection with comprehensive monitoring")
    print("=" * 80)
    
    # Initialize enhanced system
    system = EnhancedMainSystem()
    
    try:
        # Build reference database
        if not args.no_build_db:
            if not system.build_reference_database():
                print("❌ Failed to start system - reference database not built")
                return
        
        # Generate report if requested
        if args.report:
            system.generate_comprehensive_report()
        
        # Run in specified mode
        if args.mode == 'standard':
            system.run_standard_mode()
        elif args.mode == 'multi':
            system.run_multi_camera_mode()
        elif args.mode == 'api':
            system.run_api_mode()
        elif args.mode == 'batch':
            system.run_batch_mode()
        elif args.mode == 'diagnostics':
            system.run_system_diagnostics()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    finally:
        # Cleanup
        system.cleanup()
        print("\n🎉 Enhanced system session completed!")
        print("=" * 80)

if __name__ == "__main__":
    main()
