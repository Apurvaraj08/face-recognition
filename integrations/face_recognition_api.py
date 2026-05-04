#!/usr/bin/env python3
"""
Face Recognition REST API
Provides HTTP endpoints for face recognition functionality
"""

import os
import json
import time
import base64
import threading
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import io

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports")
UPLOAD_DIR = os.path.join(ROOT, "uploads")

class FaceRecognitionAPI:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize recognition system
        self.recognition_system = None
        self.alert_system = None
        self.analytics_system = None
        
        # Configure upload settings
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        self.app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
        
        # Ensure directories exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Setup routes
        self._setup_routes()
        
        # API Statistics
        self.api_stats = {
            'requests': 0,
            'recognitions': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """API health check"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'uptime': str(datetime.now() - self.api_stats['start_time'])
            })
        
        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            """Get API statistics"""
            self.api_stats['requests'] += 1
            
            return jsonify({
                'api_stats': {
                    'total_requests': self.api_stats['requests'],
                    'total_recognitions': self.api_stats['recognitions'],
                    'total_errors': self.api_stats['errors'],
                    'uptime': str(datetime.now() - self.api_stats['start_time'])
                },
                'recognition_stats': dict(self.recognition_system.recognition_stats) if self.recognition_system else {},
                'attendance_data': dict(self.recognition_system.attendance_data) if self.recognition_system else {}
            })
        
        @self.app.route('/api/recognize/image', methods=['POST'])
        def recognize_image():
            """Recognize faces in uploaded image"""
            self.api_stats['requests'] += 1
            
            try:
                # Check if image was uploaded
                if 'image' not in request.files:
                    return jsonify({'error': 'No image file provided'}), 400
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({'error': 'No image file selected'}), 400
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Read and process image
                image = cv2.imread(filepath)
                if image is None:
                    return jsonify({'error': 'Invalid image file'}), 400
                
                # Detect and recognize faces
                faces = self.recognition_system.detect_faces_stable(image)
                results = []
                
                for (x, y, w, h) in faces:
                    face_region = image[y:y+h, x:x+w]
                    name, similarity = self.recognition_system.recognize_face_stable(face_region)
                    quality = self.recognition_system.assess_face_quality_stable(face_region)
                    
                    results.append({
                        'face_id': len(results),
                        'location': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                        'recognition': {
                            'name': name,
                            'similarity': round(similarity, 3),
                            'quality': round(quality, 3)
                        }
                    })
                    
                    # Update attendance if recognized
                    if name != "Unknown":
                        self.recognition_system.update_attendance(name)
                        self.recognition_system.recognition_stats[name] += 1
                        self.api_stats['recognitions'] += 1
                
                # Clean up uploaded file
                os.remove(filepath)
                
                return jsonify({
                    'success': True,
                    'image_info': {
                        'filename': filename,
                        'dimensions': {'width': image.shape[1], 'height': image.shape[0]},
                        'faces_detected': len(faces)
                    },
                    'recognition_results': results
                })
                
            except Exception as e:
                self.api_stats['errors'] += 1
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recognize/base64', methods=['POST'])
        def recognize_base64():
            """Recognize faces in base64 encoded image"""
            self.api_stats['requests'] += 1
            
            try:
                data = request.get_json()
                if not data or 'image' not in data:
                    return jsonify({'error': 'No image data provided'}), 400
                
                # Decode base64 image
                image_data = base64.b64decode(data['image'])
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return jsonify({'error': 'Invalid image data'}), 400
                
                # Detect and recognize faces
                faces = self.recognition_system.detect_faces_stable(image)
                results = []
                
                for (x, y, w, h) in faces:
                    face_region = image[y:y+h, x:x+w]
                    name, similarity = self.recognition_system.recognize_face_stable(face_region)
                    quality = self.recognition_system.assess_face_quality_stable(face_region)
                    
                    results.append({
                        'face_id': len(results),
                        'location': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                        'recognition': {
                            'name': name,
                            'similarity': round(similarity, 3),
                            'quality': round(quality, 3)
                        }
                    })
                    
                    # Update attendance if recognized
                    if name != "Unknown":
                        self.recognition_system.update_attendance(name)
                        self.recognition_system.recognition_stats[name] += 1
                        self.api_stats['recognitions'] += 1
                
                return jsonify({
                    'success': True,
                    'image_info': {
                        'dimensions': {'width': image.shape[1], 'height': image.shape[0]},
                        'faces_detected': len(faces)
                    },
                    'recognition_results': results
                })
                
            except Exception as e:
                self.api_stats['errors'] += 1
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/students', methods=['GET'])
        def get_students():
            """Get list of recognized students"""
            self.api_stats['requests'] += 1
            
            if not self.recognition_system:
                return jsonify({'error': 'Recognition system not initialized'}), 500
            
            students = list(self.recognition_system.reference_database.keys())
            student_stats = {}
            
            for student in students:
                student_stats[student] = {
                    'total_detections': self.recognition_system.recognition_stats.get(student, 0),
                    'reference_features': len(self.recognition_system.reference_database[student]),
                    'attendance': self.recognition_system.attendance_data.get(student, {
                        'first_seen': 'N/A',
                        'last_seen': 'N/A',
                        'detection_count': 0
                    })
                }
            
            return jsonify({
                'students': student_stats,
                'total_students': len(students)
            })
        
        @self.app.route('/api/attendance', methods=['GET'])
        def get_attendance():
            """Get attendance records"""
            self.api_stats['requests'] += 1
            
            if not self.recognition_system:
                return jsonify({'error': 'Recognition system not initialized'}), 500
            
            attendance_data = {}
            for student, data in self.recognition_system.attendance_data.items():
                attendance_data[student] = {
                    'first_seen': data['first_seen'],
                    'last_seen': data['last_seen'],
                    'detection_count': data['detection_count']
                }
            
            return jsonify({
                'attendance': attendance_data,
                'total_attendance': len(attendance_data),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/reports/generate', methods=['POST'])
        def generate_report():
            """Generate and download attendance report"""
            self.api_stats['requests'] += 1
            
            try:
                data = request.get_json() or {}
                report_type = data.get('type', 'csv')
                
                if not self.recognition_system:
                    return jsonify({'error': 'Recognition system not initialized'}), 500
                
                # Generate report
                self.recognition_system.save_attendance_report()
                
                # Get latest report file
                csv_path = os.path.join(REPORTS_DIR, "stable_attendance_report.csv")
                json_path = os.path.join(REPORTS_DIR, "stable_recognition_stats.json")
                
                reports = {}
                if os.path.exists(csv_path):
                    reports['csv'] = csv_path
                if os.path.exists(json_path):
                    reports['json'] = json_path
                
                return jsonify({
                    'success': True,
                    'reports': reports,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.api_stats['errors'] += 1
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/reports/download/<filename>', methods=['GET'])
        def download_report(filename):
            """Download report file"""
            self.api_stats['requests'] += 1
            
            try:
                # Security check
                filename = secure_filename(filename)
                if not filename.endswith(('.csv', '.json')):
                    return jsonify({'error': 'Invalid file type'}), 400
                
                file_path = os.path.join(REPORTS_DIR, filename)
                if not os.path.exists(file_path):
                    return jsonify({'error': 'File not found'}), 404
                
                return send_file(file_path, as_attachment=True)
                
            except Exception as e:
                self.api_stats['errors'] += 1
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts/rules', methods=['GET', 'POST'])
        def alert_rules():
            """Get or update alert rules"""
            self.api_stats['requests'] += 1
            
            if not self.alert_system:
                return jsonify({'error': 'Alert system not initialized'}), 500
            
            if request.method == 'GET':
                return jsonify({
                    'alert_rules': self.alert_system.alert_rules
                })
            
            elif request.method == 'POST':
                try:
                    data = request.get_json()
                    if not data:
                        return jsonify({'error': 'No data provided'}), 400
                    
                    # Update alert rules
                    for rule_type, rule_config in data.items():
                        if rule_type in self.alert_system.alert_rules:
                            self.alert_system.update_alert_rule(rule_type, rule_config)
                    
                    return jsonify({
                        'success': True,
                        'alert_rules': self.alert_system.alert_rules
                    })
                    
                except Exception as e:
                    self.api_stats['errors'] += 1
                    return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts/trigger', methods=['POST'])
        def trigger_alert():
            """Manually trigger an alert"""
            self.api_stats['requests'] += 1
            
            if not self.alert_system:
                return jsonify({'error': 'Alert system not initialized'}), 500
            
            try:
                data = request.get_json()
                if not data or 'type' not in data:
                    return jsonify({'error': 'Alert type required'}), 400
                
                alert_type = data['type']
                message = data.get('message', f'Manual {alert_type} alert')
                alert_data = data.get('data', {})
                
                self.alert_system.trigger_alert(alert_type, message, alert_data)
                
                return jsonify({
                    'success': True,
                    'alert_triggered': {
                        'type': alert_type,
                        'message': message,
                        'timestamp': datetime.now().isoformat()
                    }
                })
                
            except Exception as e:
                self.api_stats['errors'] += 1
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/analytics/summary', methods=['GET'])
        def analytics_summary():
            """Get analytics summary"""
            self.api_stats['requests'] += 1
            
            if not self.analytics_system:
                return jsonify({'error': 'Analytics system not initialized'}), 500
            
            try:
                summary = self.analytics_system.get_alert_summary()
                return jsonify(summary)
                
            except Exception as e:
                self.api_stats['errors'] += 1
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/cameras/status', methods=['GET'])
        def camera_status():
            """Get camera status"""
            self.api_stats['requests'] += 1
            
            # This would integrate with multi-camera system
            return jsonify({
                'cameras': {
                    '0': {
                        'name': 'Main Camera',
                        'status': 'active',
                        'resolution': '640x480',
                        'fps': 30
                    }
                },
                'total_cameras': 1,
                'active_cameras': 1
            })
    
    def initialize_systems(self, recognition_system, alert_system=None, analytics_system=None):
        """Initialize recognition and supporting systems"""
        self.recognition_system = recognition_system
        self.alert_system = alert_system
        self.analytics_system = analytics_system
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the API server"""
        print(f"🌐 Starting Face Recognition API on http://{host}:{port}")
        print("📋 Available endpoints:")
        print("  GET  /api/health - Health check")
        print("  GET  /api/stats - API statistics")
        print("  POST /api/recognize/image - Recognize uploaded image")
        print("  POST /api/recognize/base64 - Recognize base64 image")
        print("  GET  /api/students - Get student list")
        print("  GET  /api/attendance - Get attendance records")
        print("  POST /api/reports/generate - Generate reports")
        print("  GET  /api/reports/download/<filename> - Download report")
        print("  GET/POST /api/alerts/rules - Alert rules")
        print("  POST /api/alerts/trigger - Trigger alert")
        print("  GET  /api/analytics/summary - Analytics summary")
        print("  GET  /api/cameras/status - Camera status")
        
        self.app.run(host=host, port=port, debug=debug, threaded=True)

# API Factory
def create_api_server():
    """Create and return API server instance"""
    return FaceRecognitionAPI()

# Example usage
if __name__ == "__main__":
    api = create_api_server()
    api.run()
