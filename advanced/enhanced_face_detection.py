#!/usr/bin/env python3
"""
Enhanced Face Detection System
Improved face detection with multiple algorithms and better preprocessing
"""

import os
import cv2
import numpy as np
from collections import defaultdict

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class EnhancedFaceDetector:
    def __init__(self):
        # Initialize multiple face detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.face_alt2_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Initialize facial feature detectors
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Enhanced detection parameters
        self.detection_params = {
            'scale_factor': 1.05,
            'min_neighbors': 6,
            'min_size': (40, 40),
            'max_size': (400, 400),
            'use_multi_cascade': True,
            'use_enhanced_preprocessing': True,
            'merge_overlaps': True,
            'overlap_threshold': 0.3,
            'use_adaptive_thresholding': True,
            'use_edge_enhancement': True,
            'use_histogram_equalization': True
        }
        
        # Detection statistics
        self.detection_stats = defaultdict(int)
    
    def enhanced_preprocessing(self, frame):
        """Enhanced preprocessing for better face detection"""
        processed = frame.copy()
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding for better contrast
            if self.detection_params['use_adaptive_thresholding']:
                # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray)
            else:
                gray_clahe = gray
            
            # Apply histogram equalization
            if self.detection_params['use_histogram_equalization']:
                gray_eq = cv2.equalizeHist(gray_clahe)
            else:
                gray_eq = gray_clahe
            
            # Apply Gaussian blur to reduce noise
            gray_blur = cv2.GaussianBlur(gray_eq, (3, 3), 0)
            
            # Apply bilateral filter for edge preservation
            gray_bilateral = cv2.bilateralFilter(gray_blur, 9, 75, 75)
            
            # Edge enhancement
            if self.detection_params['use_edge_enhancement']:
                edges = cv2.Canny(gray_bilateral, 50, 150)
                
                # Ensure edges has same shape for addWeighted
                if edges.shape != gray_bilateral.shape:
                    edges = cv2.resize(edges, (gray_bilateral.shape[1], gray_bilateral.shape[0]))
                
                edges_enhanced = cv2.addWeighted(gray_bilateral, 0.7, edges, 0.3, 0)
            else:
                edges_enhanced = gray_bilateral
            
            # Additional sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(edges_enhanced, -1, kernel)
            
            # Combine sharpened with original
            final_processed = cv2.addWeighted(edges_enhanced, 0.7, sharpened, 0.3, 0)
            
            return final_processed
            
        except Exception as e:
            print(f"Error in enhanced preprocessing: {e}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def detect_faces_enhanced(self, frame):
        """Enhanced face detection with multiple algorithms and preprocessing"""
        try:
            # Apply enhanced preprocessing
            if self.detection_params['use_enhanced_preprocessing']:
                processed_frame = self.enhanced_preprocessing(frame)
            else:
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            processed_frame = cv2.equalizeHist(processed_frame)
            
            all_faces = []
            
            if self.detection_params['use_multi_cascade']:
                # Use multiple cascades for better detection
                
                # Primary cascade (default) - most reliable
                faces_default = self.face_cascade.detectMultiScale(
                    processed_frame,
                    scaleFactor=self.detection_params['scale_factor'],
                    minNeighbors=self.detection_params['min_neighbors'],
                    minSize=self.detection_params['min_size'],
                    maxSize=self.detection_params['max_size']
                )
                
                # Alternative cascade - different training
                faces_alt = self.face_alt_cascade.detectMultiScale(
                    processed_frame,
                    scaleFactor=self.detection_params['scale_factor'] - 0.02,
                    minNeighbors=self.detection_params['min_neighbors'] - 1,
                    minSize=self.detection_params['min_size'],
                    maxSize=self.detection_params['max_size']
                )
                
                # Alternative 2 cascade - more sensitive
                faces_alt2 = self.face_alt2_cascade.detectMultiScale(
                    processed_frame,
                    scaleFactor=self.detection_params['scale_factor'] - 0.03,
                    minNeighbors=self.detection_params['min_neighbors'] - 2,
                    minSize=self.detection_params['min_size'],
                    maxSize=self.detection_params['max_size']
                )
                
                # Profile cascade - for side views
                faces_profile = self.profile_cascade.detectMultiScale(
                    processed_frame,
                    scaleFactor=self.detection_params['scale_factor'],
                    minNeighbors=self.detection_params['min_neighbors'] - 1,
                    minSize=self.detection_params['min_size'],
                    maxSize=self.detection_params['max_size']
                )
                
                # Combine all detections
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
            
            # Filter by size and aspect ratio
            filtered_faces = []
            for (x, y, w, h) in all_faces:
                aspect_ratio = w / h if h > 0 else 0
                face_area = w * h
                
                # Filter by size
                if (w >= self.detection_params['min_size'][0] and 
                    h >= self.detection_params['min_size'][1] and
                    w <= self.detection_params['max_size'][0] and 
                    h <= self.detection_params['max_size'][1]):
                    
                    # Filter by aspect ratio (typical face aspect ratio is 0.7 to 1.3)
                    if 0.5 < aspect_ratio < 2.0:
                        filtered_faces.append((x, y, w, h))
            
            # Additional filtering based on face quality
            quality_filtered_faces = []
            for (x, y, w, h) in filtered_faces:
                face_region = processed_frame[y:y+h, x:x+w]
                if face_region.size > 0:
                    # Simple quality check
                    face_variance = np.var(face_region)
                    if face_variance > 10:  # Minimum variance threshold
                        quality_filtered_faces.append((x, y, w, h))
            
            self.detection_stats['total_detections'] += len(quality_filtered_faces)
            
            return quality_filtered_faces
            
        except Exception as e:
            print(f"Error in enhanced face detection: {e}")
            return []
    
    def merge_overlapping_faces(self, faces):
        """Merge overlapping face detections with improved algorithm"""
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
                
                # Calculate IoU (Intersection over Union)
                intersection_x = max(0, min(x + w, mx + mw) - max(x, mx))
                intersection_y = max(0, min(y + h, my + mh) - max(y, my))
                intersection_area = intersection_x * intersection_y
                
                face_area = w * h
                merged_area = mw * mh
                union_area = face_area + merged_area - intersection_area
                
                if union_area > 0:
                    iou = intersection_area / union_area
                    if iou > self.detection_params['overlap_threshold']:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                merged_faces.append(face)
        
        # Additional merging for very close faces
        final_faces = []
        for face in merged_faces:
            x, y, w, h = face
            should_merge = False
            
            for final_face in final_faces:
                fx, fy, fw, fh = final_face
                
                # Check if centers are very close
                center_x_diff = abs((x + w/2) - (fx + fw/2))
                center_y_diff = abs((y + h/2) - (fy + fh/2))
                
                if center_x_diff < 20 and center_y_diff < 20:
                    # Merge by taking average position
                    avg_x = (x + fx) // 2
                    avg_y = (y + fy) // 2
                    avg_w = (w + fw) // 2
                    avg_h = (h + fh) // 2
                    
                    # Replace the final face with merged one
                    final_faces.remove(final_face)
                    final_faces.append((avg_x, avg_y, avg_w, avg_h))
                    should_merge = True
                    break
            
            if not should_merge:
                final_faces.append(face)
        
        return final_faces
    
    def detect_facial_landmarks_enhanced(self, face_region):
        """Enhanced facial landmark detection"""
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
            
            # Calculate enhanced landmark ratios
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
            
        except Exception as e:
            print(f"Error detecting enhanced facial landmarks: {e}")
            landmarks = {'eyes': [], 'mouth': []}
        
        return landmarks
    
    def assess_face_quality_enhanced(self, face_region):
        """Enhanced face quality assessment"""
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
            print(f"Error in enhanced quality assessment: {e}")
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
