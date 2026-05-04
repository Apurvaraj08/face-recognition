#!/usr/bin/env python3
"""
Advanced Face Detection System
Improved face detection with advanced preprocessing and facial landmarks
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime
from collections import defaultdict

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

class AdvancedFaceDetector:
    def __init__(self):
        # Initialize multiple face detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.face_alt2_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Initialize facial feature detectors
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Nose cascade (optional - may not be available in all OpenCV installations)
        try:
            self.nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
            if self.nose_cascade.empty():
                self.nose_cascade = None
        except:
            self.nose_cascade = None
        
        # Detection parameters
        self.detection_params = {
            'scale_factor': 1.05,
            'min_neighbors': 6,
            'min_size': (40, 40),
            'max_size': (400, 400),
            'use_multi_cascade': True,
            'use_advanced_preprocessing': True,
            'merge_overlaps': True,
            'overlap_threshold': 0.3
        }
        
        # Detection statistics
        self.detection_stats = defaultdict(int)
    
    def advanced_preprocessing(self, frame):
        """Advanced preprocessing for better face detection"""
        processed = frame.copy()
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            gray_eq = cv2.equalizeHist(gray)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_clahe = clahe.apply(gray_eq)
            
            # Apply Gaussian blur to reduce noise
            gray_blur = cv2.GaussianBlur(gray_clahe, (3, 3), 0)
            
            # Apply bilateral filter for edge preservation
            gray_bilateral = cv2.bilateralFilter(gray_blur, 9, 75, 75)
            
            # Enhance edges
            edges = cv2.Canny(gray_bilateral, 50, 150)
            
            # Ensure edges has same shape as gray_bilateral for addWeighted
            if edges.shape != gray_bilateral.shape:
                edges = cv2.resize(edges, (gray_bilateral.shape[1], gray_bilateral.shape[0]))
            
            edges_enhanced = cv2.addWeighted(gray_bilateral, 0.7, edges, 0.3, 0)
            
            return edges_enhanced
            
        except Exception as e:
            print(f"Error in advanced preprocessing: {e}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def detect_faces_advanced(self, frame):
        """Advanced face detection with multiple cascades and preprocessing"""
        try:
            # Apply advanced preprocessing
            if self.detection_params['use_advanced_preprocessing']:
                processed_frame = self.advanced_preprocessing(frame)
            else:
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            processed_frame = cv2.equalizeHist(processed_frame)
            
            all_faces = []
            
            if self.detection_params['use_multi_cascade']:
                # Use multiple cascades for better detection
                faces_default = self.face_cascade.detectMultiScale(
                    processed_frame,
                    scaleFactor=self.detection_params['scale_factor'],
                    minNeighbors=self.detection_params['min_neighbors'],
                    minSize=self.detection_params['min_size'],
                    maxSize=self.detection_params['max_size']
                )
                
                faces_alt = self.face_alt_cascade.detectMultiScale(
                    processed_frame,
                    scaleFactor=self.detection_params['scale_factor'],
                    minNeighbors=self.detection_params['min_neighbors'] - 1,
                    minSize=self.detection_params['min_size'],
                    maxSize=self.detection_params['max_size']
                )
                
                faces_alt2 = self.face_alt2_cascade.detectMultiScale(
                    processed_frame,
                    scaleFactor=self.detection_params['scale_factor'],
                    minNeighbors=self.detection_params['min_neighbors'] - 2,
                    minSize=self.detection_params['min_size'],
                    maxSize=self.detection_params['max_size']
                )
                
                faces_profile = self.profile_cascade.detectMultiScale(
                    processed_frame,
                    scaleFactor=self.detection_params['scale_factor'],
                    minNeighbors=self.detection_params['min_neighbors'] - 1,
                    minSize=self.detection_params['min_size'],
                    maxSize=self.detection_params['max_size']
                )
                
                all_faces = list(faces_default) + list(faces_alt) + list(faces_alt2) + list(faces_profile)
                
                # Update statistics
                self.detection_stats['default_cascade'] += len(faces_default)
                self.detection_stats['alt_cascade'] += len(faces_alt)
                self.detection_stats['alt2_cascade'] += len(faces_alt2)
                self.detection_stats['profile_cascade'] += len(faces_profile)
            else:
                # Use only default cascade
                all_faces = self.face_cascade.detectMultiScale(
                    processed_frame,
                    scaleFactor=self.detection_params['scale_factor'],
                    minNeighbors=self.detection_params['min_neighbors'],
                    minSize=self.detection_params['min_size'],
                    maxSize=self.detection_params['max_size']
                )
            
            # Merge overlapping detections
            if self.detection_params['merge_overlaps']:
                all_faces = self.merge_overlapping_faces(all_faces)
            
            # Filter by size
            filtered_faces = []
            for (x, y, w, h) in all_faces:
                if (w >= self.detection_params['min_size'][0] and 
                    h >= self.detection_params['min_size'][1] and
                    w <= self.detection_params['max_size'][0] and 
                    h <= self.detection_params['max_size'][1]):
                    filtered_faces.append((x, y, w, h))
            
            self.detection_stats['total_detections'] += len(filtered_faces)
            
            return filtered_faces
            
        except Exception as e:
            print(f"Error in advanced face detection: {e}")
            return []
    
    def merge_overlapping_faces(self, faces):
        """Merge overlapping face detections"""
        if not faces:
            return []
        
        # Sort faces by size (largest first)
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        
        merged_faces = []
        
        for face in faces_sorted:
            x, y, w, h = face
            is_duplicate = False
            
            for merged_face in merged_faces:
                mx, my, mw, mh = merged_face
                
                # Calculate overlap
                overlap_x = max(0, min(x + w, mx + mw) - max(x, mx))
                overlap_y = max(0, min(y + h, my + mh) - max(y, my))
                overlap_area = overlap_x * overlap_y
                
                face_area = w * h
                merged_area = mw * mh
                union_area = face_area + merged_area - overlap_area
                
                if union_area > 0:
                    iou = overlap_area / union_area
                    if iou > self.detection_params['overlap_threshold']:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                merged_faces.append(face)
        
        return merged_faces
    
    def detect_facial_landmarks(self, face_region):
        """Detect facial landmarks with improved accuracy"""
        landmarks = {}
        
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Enhanced eye detection
            eyes = self.eye_cascade.detectMultiScale(gray, 1.05, 5, minSize=(20, 20))
            landmarks['eyes'] = []
            for (ex, ey, ew, eh) in eyes:
                # Filter eye candidates by aspect ratio
                aspect_ratio = ew / eh if eh > 0 else 0
                if 0.5 < aspect_ratio < 2.5:  # Typical eye aspect ratio
                    landmarks['eyes'].append({
                        'x': int(ex), 'y': int(ey),
                        'width': int(ew), 'height': int(eh),
                        'center': (int(ex + ew/2), int(ey + eh/2)),
                        'aspect_ratio': aspect_ratio
                    })
            
            # Sort eyes by x-coordinate (left to right)
            landmarks['eyes'].sort(key=lambda x: x['x'])
            
            # Enhanced mouth detection
            mouth_region = gray[int(height*0.6):, :]  # Lower 40% of face
            mouth = self.mouth_cascade.detectMultiScale(mouth_region, 1.3, 15, minSize=(30, 10))
            landmarks['mouth'] = []
            for (mx, my, mw, mh) in mouth:
                # Adjust coordinates to full face region
                my += int(height*0.6)
                # Filter mouth candidates
                if mw > 30 and mh < 40:  # Typical mouth dimensions
                    landmarks['mouth'].append({
                        'x': int(mx), 'y': int(my),
                        'width': int(mw), 'height': int(mh),
                        'center': (int(mx + mw/2), int(my + mh/2)),
                        'aspect_ratio': mw/mh if mh > 0 else 0
                    })
            
            # Enhanced nose detection (if available)
            if self.nose_cascade is not None:
                nose_region = gray[int(height*0.3):int(height*0.7), int(width*0.3):int(width*0.7)]
                if nose_region.size > 0:
                    nose = self.nose_cascade.detectMultiScale(nose_region, 1.1, 3)
                    landmarks['nose'] = []
                    for (nx, ny, nw, nh) in nose:
                        # Adjust coordinates to full face region
                        nx += int(width*0.3)
                        ny += int(height*0.3)
                        # Filter nose candidates
                        if 20 < nw < 60 and 20 < nh < 80:
                            landmarks['nose'].append({
                                'x': int(nx), 'y': int(ny),
                                'width': int(nw), 'height': int(nh),
                                'center': (int(nx + nw/2), int(ny + nh/2)),
                                'aspect_ratio': nw/nh if nh > 0 else 0
                            })
            else:
                landmarks['nose'] = []
            
            # Calculate improved landmark ratios and features
            if len(landmarks['eyes']) >= 2:
                eye1 = landmarks['eyes'][0]
                eye2 = landmarks['eyes'][1]
                
                # Eye distance and ratio
                eye_distance = np.sqrt((eye1['center'][0] - eye2['center'][0])**2 + 
                                      (eye1['center'][1] - eye2['center'][1])**2)
                landmarks['eye_distance'] = eye_distance
                landmarks['eye_ratio'] = eye_distance / width
                
                # Eye symmetry
                eye1_y = eye1['center'][1]
                eye2_y = eye2['center'][1]
                landmarks['eye_symmetry'] = abs(eye1_y - eye2_y) / height
                
                # Eye sizes
                eye1_size = eye1['width'] * eye1['height']
                eye2_size = eye2['width'] * eye2['height']
                landmarks['eye_size_ratio'] = min(eye1_size, eye2_size) / max(eye1_size, eye2_size)
            
            if landmarks['mouth']:
                mouth = landmarks['mouth'][0]
                landmarks['mouth_ratio'] = mouth['width'] / width
                landmarks['mouth_position'] = mouth['center'][1] / height
                
                # Mouth curvature (simplified)
                mouth_bottom = mouth['y'] + mouth['height']
                mouth_top = mouth['y']
                landmarks['mouth_height_ratio'] = (mouth_bottom - mouth_top) / height
            
            if landmarks['nose']:
                nose = landmarks['nose'][0]
                landmarks['nose_ratio'] = nose['width'] / width
                landmarks['nose_position'] = nose['center'][1] / height
            
            # Face proportions
            if len(landmarks['eyes']) >= 2 and landmarks['nose']:
                # Eye-nose distance
                eye_center_y = (landmarks['eyes'][0]['center'][1] + landmarks['eyes'][1]['center'][1]) / 2
                nose_center_y = landmarks['nose'][0]['center'][1]
                landmarks['eye_nose_distance'] = (nose_center_y - eye_center_y) / height
                
                # Nose-mouth distance
                if landmarks['mouth']:
                    mouth_center_y = landmarks['mouth'][0]['center'][1]
                    landmarks['nose_mouth_distance'] = (mouth_center_y - nose_center_y) / height
            
        except Exception as e:
            print(f"Error detecting facial landmarks: {e}")
            # Return empty landmarks on error
            landmarks = {'eyes': [], 'mouth': [], 'nose': []}
        
        return landmarks
    
    def estimate_face_pose(self, face_region, landmarks):
        """Estimate face pose (yaw, pitch, roll) with improved accuracy"""
        pose = {'yaw': 0, 'pitch': 0, 'roll': 0}
        
        try:
            height, width = face_region.shape[:2]
            
            # Estimate roll from eye angle
            if len(landmarks.get('eyes', [])) >= 2:
                eye1 = landmarks['eyes'][0]
                eye2 = landmarks['eyes'][1]
                
                # Use eye centers for more accurate angle calculation
                eye1_center = eye1['center']
                eye2_center = eye2['center']
                
                dx = eye2_center[0] - eye1_center[0]
                dy = eye2_center[1] - eye1_center[1]
                
                if dx != 0:
                    angle = np.degrees(np.arctan2(dy, dx))
                    pose['roll'] = angle
                
                # Estimate yaw from eye asymmetry
                face_center_x = width / 2
                eyes_center_x = (eye1_center[0] + eye2_center[0]) / 2
                
                yaw_offset = (eyes_center_x - face_center_x) / face_center_x
                pose['yaw'] = np.clip(yaw_offset * 30, -45, 45)  # Scale and clamp to degrees
            
            # Estimate pitch from facial feature positions
            if landmarks.get('eyes') and landmarks.get('nose'):
                eye_center_y = (landmarks['eyes'][0]['center'][1] + landmarks['eyes'][1]['center'][1]) / 2
                nose_center_y = landmarks['nose'][0]['center'][1]
                face_center_y = height / 2
                
                # Use eye-nose distance for pitch estimation
                eye_nose_distance = nose_center_y - eye_center_y
                expected_distance = height * 0.15  # Expected eye-nose distance
                
                if expected_distance > 0:
                    pitch_offset = (eye_nose_distance - expected_distance) / expected_distance
                    pose['pitch'] = np.clip(pitch_offset * 20, -30, 30)  # Scale and clamp to degrees
            
            # Additional pitch estimation using mouth position
            if landmarks.get('mouth') and landmarks.get('nose'):
                nose_center_y = landmarks['nose'][0]['center'][1]
                mouth_center_y = landmarks['mouth'][0]['center'][1]
                nose_mouth_distance = mouth_center_y - nose_center_y
                expected_nose_mouth = height * 0.25  # Expected nose-mouth distance
                
                if expected_nose_mouth > 0:
                    mouth_pitch_offset = (nose_mouth_distance - expected_nose_mouth) / expected_nose_mouth
                    mouth_pitch = np.clip(mouth_pitch_offset * 15, -20, 20)
                    # Average with previous pitch estimate if available
                    if pose['pitch'] != 0:
                        pose['pitch'] = (pose['pitch'] + mouth_pitch) / 2
                    else:
                        pose['pitch'] = mouth_pitch
            
        except Exception as e:
            print(f"Error estimating face pose: {e}")
            # Return neutral pose on error
            pose = {'yaw': 0, 'pitch': 0, 'roll': 0}
        
        return pose
    
    def detect_age_gender(self, face_region):
        """Estimate age and gender from face features (rule-based)"""
        age_gender = {
            'age_range': 'Unknown',
            'gender': 'Unknown',
            'confidence': 0.0
        }
        
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Extract features for age/gender estimation
            # This is a simplified rule-based approach
            # In production, you would use trained models
            
            # Skin texture analysis
            texture_variance = np.var(gray)
            
            # Wrinkle detection (edge density)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            # Face shape analysis
            height, width = gray.shape
            aspect_ratio = height / width
            
            # Simple age estimation based on texture and edges
            if texture_variance < 50 and edge_density < 0.1:
                age_gender['age_range'] = 'Young (18-30)'
                age_gender['confidence'] = 0.6
            elif texture_variance < 100 and edge_density < 0.15:
                age_gender['age_range'] = 'Adult (30-50)'
                age_gender['confidence'] = 0.5
            else:
                age_gender['age_range'] = 'Senior (50+)'
                age_gender['confidence'] = 0.4
            
            # Simple gender estimation based on face shape
            if aspect_ratio > 1.3:
                age_gender['gender'] = 'Male'
                age_gender['confidence'] = 0.5
            elif aspect_ratio < 1.1:
                age_gender['gender'] = 'Female'
                age_gender['confidence'] = 0.5
            else:
                age_gender['gender'] = 'Unknown'
                age_gender['confidence'] = 0.3
            
        except Exception as e:
            print(f"Error detecting age/gender: {e}")
        
        return age_gender
    
    def assess_face_quality_advanced(self, face_region):
        """Advanced face quality assessment with multiple metrics"""
        quality_metrics = {}
        
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # 1. Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_metrics['sharpness'] = min(1.0, sharpness / 100)
            
            # 2. Brightness
            brightness = np.mean(gray)
            quality_metrics['brightness'] = 1.0 - abs(brightness - 128) / 128
            
            # 3. Contrast
            contrast = np.std(gray)
            quality_metrics['contrast'] = min(1.0, contrast / 80)
            
            # 4. Face size
            face_area = height * width
            quality_metrics['size'] = min(1.0, face_area / (100 * 100))
            
            # 5. Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            quality_metrics['edge_density'] = min(1.0, edge_density * 10)
            
            # 6. Symmetry
            left_half = gray[:, :width//2]
            right_half = np.fliplr(gray[:, width//2:])
            if left_half.shape == right_half.shape:
                correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
                quality_metrics['symmetry'] = max(0, correlation) if not np.isnan(correlation) else 0.5
            else:
                quality_metrics['symmetry'] = 0.5
            
            # 7. Eye detection quality
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)
            quality_metrics['eye_detection'] = min(1.0, len(eyes) / 2)
            
            # 8. Face aspect ratio
            aspect_ratio = height / width
            ideal_ratio = 1.3
            quality_metrics['aspect_ratio'] = 1.0 - abs(aspect_ratio - ideal_ratio) / ideal_ratio
            
            # 9. Histogram distribution
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_normalized = hist / np.sum(hist)
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-7))
            quality_metrics['entropy'] = min(1.0, entropy / 8)
            
            # 10. Overall quality score (weighted average)
            weights = {
                'sharpness': 0.2,
                'brightness': 0.15,
                'contrast': 0.15,
                'size': 0.1,
                'edge_density': 0.1,
                'symmetry': 0.1,
                'eye_detection': 0.1,
                'aspect_ratio': 0.05,
                'entropy': 0.05
            }
            
            overall_quality = sum(quality_metrics[metric] * weights[metric] 
                                 for metric in weights)
            quality_metrics['overall'] = max(0, min(1, overall_quality))
            
        except Exception as e:
            print(f"Error in advanced quality assessment: {e}")
            quality_metrics = {'overall': 0}
        
        return quality_metrics
    
    def get_detection_statistics(self):
        """Get detection statistics"""
        return dict(self.detection_stats)
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.detection_stats.clear()
    
    def update_detection_params(self, params):
        """Update detection parameters"""
        for key, value in params.items():
            if key in self.detection_params:
                self.detection_params[key] = value
