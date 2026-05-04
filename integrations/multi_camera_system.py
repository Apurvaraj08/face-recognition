#!/usr/bin/env python3
"""
Multi-Camera Face Recognition System
Supports multiple camera feeds simultaneously
"""

import os
import cv2
import numpy as np
import json
import time
import threading
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import queue

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports")
CONFIG_DIR = os.path.join(ROOT, "config")

class MultiCameraSystem:
    def __init__(self):
        self.cameras = {}
        self.camera_threads = {}
        self.recognition_system = None
        self.running = False
        self.frame_queues = {}
        self.result_queue = queue.Queue()
        self.camera_stats = defaultdict(lambda: defaultdict(int))
        
        # Load camera configuration
        self.camera_config = self.load_camera_config()
        
    def load_camera_config(self):
        """Load camera configuration"""
        config_file = os.path.join(CONFIG_DIR, "camera_config.json")
        default_config = {
            "cameras": [
                {
                    "id": 0,
                    "name": "Main Camera",
                    "resolution": [640, 480],
                    "fps": 30,
                    "enabled": True,
                    "position": "entrance"
                },
                {
                    "id": 1,
                    "name": "Secondary Camera", 
                    "resolution": [640, 480],
                    "fps": 30,
                    "enabled": False,
                    "position": "hallway"
                },
                {
                    "id": 2,
                    "name": "Third Camera",
                    "resolution": [320, 240],
                    "fps": 15,
                    "enabled": False,
                    "position": "classroom"
                }
            ],
            "recognition": {
                "process_every_n_frames": 3,
                "max_concurrent_recognitions": 2,
                "sync_cameras": True
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                for key, value in loaded_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            except:
                pass
        
        return default_config
    
    def initialize_cameras(self):
        """Initialize all configured cameras"""
        print("=== Initializing Multi-Camera System ===")
        
        initialized_cameras = 0
        
        for camera_config in self.camera_config["cameras"]:
            if not camera_config.get("enabled", False):
                print(f"⏸️  Camera {camera_config['id']} ({camera_config['name']}) - Disabled")
                continue
            
            camera_id = camera_config["id"]
            camera_name = camera_config["name"]
            
            # Initialize camera
            cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                # Set camera properties
                width, height = camera_config.get("resolution", [640, 480])
                fps = camera_config.get("fps", 30)
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                
                # Test camera
                ret, frame = cap.read()
                if ret:
                    self.cameras[camera_id] = {
                        'capture': cap,
                        'config': camera_config,
                        'last_frame': frame,
                        'frame_count': 0,
                        'last_recognition': 0
                    }
                    
                    # Create frame queue for this camera
                    self.frame_queues[camera_id] = queue.Queue(maxsize=10)
                    
                    print(f"✅ Camera {camera_id} ({camera_name}) - Initialized ({width}x{height} @ {fps}fps)")
                    initialized_cameras += 1
                else:
                    print(f"❌ Camera {camera_id} ({camera_name}) - Failed to capture frame")
                    cap.release()
            else:
                print(f"❌ Camera {camera_id} ({camera_name}) - Not available")
        
        print(f"📷 Initialized {initialized_cameras} cameras")
        return initialized_cameras > 0
    
    def start_multi_camera_recognition(self, recognition_system):
        """Start multi-camera face recognition"""
        self.recognition_system = recognition_system
        self.running = True
        
        if not self.cameras:
            print("❌ No cameras available")
            return False
        
        print(f"🚀 Starting multi-camera recognition with {len(self.cameras)} cameras")
        
        # Start camera threads
        for camera_id in self.cameras:
            thread = threading.Thread(
                target=self._camera_worker,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            self.camera_threads[camera_id] = thread
        
        # Start recognition processor
        recognition_thread = threading.Thread(
            target=self._recognition_processor,
            daemon=True
        )
        recognition_thread.start()
        
        # Start display thread
        display_thread = threading.Thread(
            target=self._display_worker,
            daemon=True
        )
        display_thread.start()
        
        return True
    
    def _camera_worker(self, camera_id):
        """Worker thread for capturing frames from a camera"""
        camera = self.cameras[camera_id]
        cap = camera['capture']
        frame_queue = self.frame_queues[camera_id]
        
        process_every_n = self.camera_config["recognition"]["process_every_n_frames"]
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Camera {camera_id} - Failed to read frame")
                time.sleep(0.1)
                continue
            
            camera['frame_count'] += 1
            camera['last_frame'] = frame.copy()
            
            # Add frame to queue for processing
            if camera['frame_count'] % process_every_n == 0:
                try:
                    frame_queue.put({
                        'camera_id': camera_id,
                        'frame': frame,
                        'timestamp': datetime.now(),
                        'frame_count': camera['frame_count']
                    }, timeout=0.1)
                except queue.Full:
                    # Drop frame if queue is full
                    pass
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.03)
    
    def _recognition_processor(self):
        """Process frames from all cameras for face recognition"""
        max_concurrent = self.camera_config["recognition"]["max_concurrent_recognitions"]
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            while self.running:
                # Get frames from all cameras
                frames_to_process = []
                
                for camera_id, frame_queue in self.frame_queues.items():
                    try:
                        frame_data = frame_queue.get_nowait()
                        frames_to_process.append(frame_data)
                    except queue.Empty:
                        continue
                
                if frames_to_process:
                    # Process frames concurrently
                    futures = []
                    for frame_data in frames_to_process:
                        future = executor.submit(
                            self._process_frame_recognition,
                            frame_data
                        )
                        futures.append(future)
                    
                    # Collect results
                    for future in futures:
                        try:
                            result = future.result(timeout=1.0)
                            if result:
                                self.result_queue.put(result)
                        except:
                            pass
                
                time.sleep(0.1)
    
    def _process_frame_recognition(self, frame_data):
        """Process a single frame for face recognition"""
        camera_id = frame_data['camera_id']
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        
        try:
            # Detect faces
            faces = self.recognition_system.detect_faces_stable(frame)
            
            results = []
            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                
                # Recognize face
                name, similarity = self.recognition_system.recognize_face_stable(face_region)
                
                if name != "Unknown":
                    # Update attendance
                    self.recognition_system.update_attendance(name)
                    self.recognition_system.recognition_stats[name] += 1
                    
                    # Update camera stats
                    self.camera_stats[camera_id][name] += 1
                    self.camera_stats[camera_id]['total_detections'] += 1
                
                results.append({
                    'camera_id': camera_id,
                    'timestamp': timestamp,
                    'face_location': (x, y, w, h),
                    'recognition': {
                        'name': name,
                        'similarity': similarity
                    }
                })
            
            return {
                'camera_id': camera_id,
                'timestamp': timestamp,
                'frame': frame,
                'faces': results
            }
            
        except Exception as e:
            print(f"Error processing frame from camera {camera_id}: {e}")
            return None
    
    def _display_worker(self):
        """Display frames from all cameras"""
        sync_cameras = self.camera_config["recognition"]["sync_cameras"]
        
        while self.running:
            try:
                # Collect latest frames from all cameras
                frames_to_display = {}
                
                for camera_id, camera in self.cameras.items():
                    if camera['last_frame'] is not None:
                        frames_to_display[camera_id] = camera['last_frame'].copy()
                
                if frames_to_display:
                    if sync_cameras and len(frames_to_display) > 1:
                        # Display all cameras in a grid
                        self._display_camera_grid(frames_to_display)
                    else:
                        # Display cameras separately
                        self._display_cameras_separately(frames_to_display)
                
                time.sleep(0.05)  # ~20 FPS display
                
            except Exception as e:
                print(f"Error in display worker: {e}")
                time.sleep(0.1)
    
    def _display_camera_grid(self, frames_to_display):
        """Display multiple cameras in a grid layout"""
        camera_count = len(frames_to_display)
        
        if camera_count == 0:
            return
        
        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(camera_count)))
        rows = int(np.ceil(camera_count / cols))
        
        # Create grid image
        max_width = max(frame.shape[1] for frame in frames_to_display.values())
        max_height = max(frame.shape[0] for frame in frames_to_display.values())
        
        grid_image = np.zeros((rows * max_height, cols * max_width, 3), dtype=np.uint8)
        
        for idx, (camera_id, frame) in enumerate(frames_to_display.items()):
            row = idx // cols
            col = idx % cols
            
            # Resize frame to fit grid
            resized_frame = cv2.resize(frame, (max_width, max_height))
            
            # Add camera info overlay
            camera_config = self.cameras[camera_id]['config']
            camera_name = camera_config['name']
            position = camera_config.get('position', 'unknown')
            
            cv2.putText(resized_frame, f"{camera_name} ({position})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add camera stats
            stats = self.camera_stats[camera_id]
            total_detections = stats.get('total_detections', 0)
            cv2.putText(resized_frame, f"Detections: {total_detections}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Place in grid
            y_start = row * max_height
            y_end = y_start + max_height
            x_start = col * max_width
            x_end = x_start + max_width
            
            grid_image[y_start:y_end, x_start:x_end] = resized_frame
        
        # Add grid info
        cv2.putText(grid_image, f"Multi-Camera System - {camera_count} Cameras", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Multi-Camera Face Recognition', grid_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
        elif key == ord('s'):
            self.save_multi_camera_report()
    
    def _display_cameras_separately(self, frames_to_display):
        """Display cameras in separate windows"""
        for camera_id, frame in frames_to_display.items():
            window_name = f"Camera {camera_id}"
            
            # Add camera info
            camera_config = self.cameras[camera_id]['config']
            camera_name = camera_config['name']
            position = camera_config.get('position', 'unknown')
            
            display_frame = frame.copy()
            cv2.putText(display_frame, f"{camera_name} ({position})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add stats
            stats = self.camera_stats[camera_id]
            total_detections = stats.get('total_detections', 0)
            cv2.putText(display_frame, f"Detections: {total_detections}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
        elif key == ord('s'):
            self.save_multi_camera_report()
    
    def stop_multi_camera_system(self):
        """Stop the multi-camera system"""
        print("🛑 Stopping multi-camera system...")
        
        self.running = False
        
        # Wait for threads to finish
        for thread in self.camera_threads.values():
            if thread.is_alive():
                thread.join(timeout=2)
        
        # Release cameras
        for camera_id, camera in self.cameras.items():
            camera['capture'].release()
        
        cv2.destroyAllWindows()
        print("✅ Multi-camera system stopped")
    
    def save_multi_camera_report(self):
        """Save multi-camera recognition report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = {
            'timestamp': timestamp,
            'cameras_used': list(self.cameras.keys()),
            'camera_stats': dict(self.camera_stats),
            'total_detections': sum(stats.get('total_detections', 0) for stats in self.camera_stats.values()),
            'recognition_stats': dict(self.recognition_system.recognition_stats),
            'attendance_data': dict(self.recognition_system.attendance_data)
        }
        
        # Save JSON report
        json_path = os.path.join(REPORTS_DIR, f"multi_camera_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Multi-camera report saved to: {json_path}")
        
        # Display summary
        print(f"\n📊 Multi-Camera Recognition Summary ({timestamp}):")
        print("-" * 60)
        print(f"Cameras Active: {len(self.cameras)}")
        print(f"Total Detections: {report['total_detections']}")
        
        for camera_id, stats in self.camera_stats.items():
            camera_name = self.cameras[camera_id]['config']['name']
            detections = stats.get('total_detections', 0)
            print(f"  {camera_name}: {detections} detections")
        
        print("-" * 60)
        
        return json_path
    
    def get_camera_status(self):
        """Get status of all cameras"""
        status = {}
        for camera_id, camera in self.cameras.items():
            config = camera['config']
            stats = self.camera_stats[camera_id]
            
            status[camera_id] = {
                'name': config['name'],
                'position': config.get('position', 'unknown'),
                'resolution': config.get('resolution', [640, 480]),
                'fps': config.get('fps', 30),
                'frame_count': camera['frame_count'],
                'total_detections': stats.get('total_detections', 0),
                'student_detections': {k: v for k, v in stats.items() if k != 'total_detections'}
            }
        
        return status

# Integration with main recognition system
def run_multi_camera_system(recognition_system):
    """Run multi-camera system with recognition"""
    multi_cam = MultiCameraSystem()
    
    try:
        if multi_cam.initialize_cameras():
            if multi_cam.start_multi_camera_recognition(recognition_system):
                print("🎥 Multi-camera system running. Press 'q' to quit, 's' to save report.")
                
                # Keep running until stopped
                while multi_cam.running:
                    time.sleep(1)
            else:
                print("❌ Failed to start multi-camera recognition")
        else:
            print("❌ No cameras initialized")
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        multi_cam.stop_multi_camera_system()

if __name__ == "__main__":
    # Example usage (would need recognition_system instance)
    print("Multi-camera system requires integration with main recognition system")
    print("Use run_multi_camera_system(recognition_system) from main application")
