#!/usr/bin/env python3
"""
Face Tracking System with Kalman Filter
Tracks faces across frames for better recognition continuity
"""

import os
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime

class FaceTracker:
    def __init__(self, max_age=30, min_hits=3):
        self.max_age = max_age  # Maximum frames to keep track without detection
        self.min_hits = min_hits  # Minimum detections before confirming track
        self.tracks = []
        self.track_id_counter = 0
        self.frame_count = 0
        
        # Tracking statistics
        self.tracking_stats = defaultdict(int)
    
    def update(self, detections):
        """Update tracks with new detections"""
        self.frame_count += 1
        
        # Predict new positions for existing tracks
        for track in self.tracks:
            track['kalman'].predict()
        
        # Associate detections with tracks
        if detections:
            self._associate_detections(detections)
        
        # Update track ages
        for track in self.tracks:
            track['age'] += 1
            if track['hits'] >= self.min_hits:
                track['confirmed'] = True
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks 
                      if track['age'] < self.max_age or track['confirmed']]
        
        # Return confirmed tracks
        confirmed_tracks = [track for track in self.tracks if track['confirmed']]
        self.tracking_stats['confirmed_tracks'] = len(confirmed_tracks)
        self.tracking_stats['total_tracks'] = len(self.tracks)
        
        return confirmed_tracks
    
    def _associate_detections(self, detections):
        """Associate new detections with existing tracks"""
        if not self.tracks:
            # Create new tracks for all detections
            for detection in detections:
                self._create_track(detection)
            return
        
        # Calculate cost matrix (IoU)
        cost_matrix = self._calculate_cost_matrix(detections)
        
        # Hungarian algorithm for optimal assignment
        rows, cols = self._linear_assignment(cost_matrix)
        
        # Update matched tracks
        for row, col in zip(rows, cols):
            if cost_matrix[row, col] < 0.5:  # IoU threshold
                self._update_track(self.tracks[row], detections[col])
                detections[col]['matched'] = True
            else:
                # Create new track for unmatched detection
                if not detections[col].get('matched', False):
                    self._create_track(detections[col])
        
        # Create new tracks for unmatched detections
        for detection in detections:
            if not detection.get('matched', False):
                self._create_track(detection)
    
    def _calculate_cost_matrix(self, detections):
        """Calculate cost matrix based on IoU"""
        num_tracks = len(self.tracks)
        num_detections = len(detections)
        
        cost_matrix = np.ones((num_tracks, num_detections))
        
        for i, track in enumerate(self.tracks):
            track_bbox = track['bbox']
            for j, detection in enumerate(detections):
                detection_bbox = detection
                iou = self._calculate_iou(track_bbox, detection_bbox)
                cost_matrix[i, j] = 1 - iou  # Cost = 1 - IoU
        
        return cost_matrix
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU)"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _linear_assignment(self, cost_matrix):
        """Solve linear assignment problem (simplified Hungarian algorithm)"""
        # Simplified version - for production use scipy.optimize.linear_sum_assignment
        rows = []
        cols = []
        
        # Greedy assignment (not optimal but fast)
        for i in range(cost_matrix.shape[0]):
            min_idx = np.argmin(cost_matrix[i])
            if min_idx < cost_matrix.shape[1]:
                rows.append(i)
                cols.append(min_idx)
        
        return rows, cols
    
    def _create_track(self, detection):
        """Create a new track from detection"""
        x, y, w, h = detection
        
        # Initialize Kalman filter
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0], [0, 1]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Initialize state
        kalman.statePre = np.array([x + w/2, y + h/2, 0, 0], dtype=np.float32)
        kalman.statePost = np.array([x + w/2, y + h/2, 0, 0], dtype=np.float32)
        
        track = {
            'id': self.track_id_counter,
            'kalman': kalman,
            'bbox': (x, y, w, h),
            'age': 0,
            'hits': 1,
            'confirmed': False,
            'first_seen': self.frame_count,
            'last_seen': self.frame_count
        }
        
        self.tracks.append(track)
        self.track_id_counter += 1
        self.tracking_stats['tracks_created'] += 1
    
    def _update_track(self, track, detection):
        """Update existing track with new detection"""
        x, y, w, h = detection
        
        # Update Kalman filter with measurement
        measurement = np.array([x + w/2, y + h/2], dtype=np.float32)
        track['kalman'].correct(measurement)
        
        # Update track
        track['bbox'] = (x, y, w, h)
        track['hits'] += 1
        track['last_seen'] = self.frame_count
        track['age'] = 0  # Reset age when detected
        
        self.tracking_stats['tracks_updated'] += 1
    
    def get_track_info(self, track_id):
        """Get information about a specific track"""
        for track in self.tracks:
            if track['id'] == track_id:
                return {
                    'id': track['id'],
                    'bbox': track['bbox'],
                    'age': track['age'],
                    'hits': track['hits'],
                    'confirmed': track['confirmed'],
                    'duration': track['last_seen'] - track['first_seen']
                }
        return None
    
    def get_tracking_statistics(self):
        """Get tracking statistics"""
        return dict(self.tracking_stats)
    
    def reset(self):
        """Reset all tracks"""
        self.tracks = []
        self.track_id_counter = 0
        self.frame_count = 0
        self.tracking_stats.clear()

class AdvancedFaceTracker(FaceTracker):
    """Advanced face tracker with additional features"""
    
    def __init__(self, max_age=30, min_hits=3):
        super().__init__(max_age, min_hits)
        self.track_history = defaultdict(list)
        self.recognition_history = defaultdict(list)
    
    def update_with_recognition(self, detections, recognitions):
        """Update tracks with both detections and recognitions"""
        confirmed_tracks = self.update(detections)
        
        # Associate recognitions with tracks
        for track in confirmed_tracks:
            track_id = track['id']
            track_bbox = track['bbox']
            
            # Find matching recognition
            for detection, recognition in zip(detections, recognitions):
                if self._calculate_iou(track_bbox, detection) > 0.5:
                    # Store recognition with track
                    self.recognition_history[track_id].append({
                        'recognition': recognition,
                        'timestamp': self.frame_count
                    })
                    
                    # Store track position
                    self.track_history[track_id].append({
                        'bbox': track_bbox,
                        'timestamp': self.frame_count
                    })
                    
                    # Keep only recent history
                    if len(self.track_history[track_id]) > 100:
                        self.track_history[track_id].pop(0)
                    if len(self.recognition_history[track_id]) > 50:
                        self.recognition_history[track_id].pop(0)
        
        return confirmed_tracks
    
    def get_track_recognition_summary(self, track_id):
        """Get recognition summary for a track"""
        if track_id not in self.recognition_history:
            return None
        
        recognitions = self.recognition_history[track_id]
        
        if not recognitions:
            return None
        
        # Count recognitions
        recognition_counts = defaultdict(int)
        for rec in recognitions:
            name = rec['recognition'].get('name', 'Unknown')
            recognition_counts[name] += 1
        
        # Get most common recognition
        most_common = max(recognition_counts, key=recognition_counts.get)
        confidence = recognition_counts[most_common] / len(recognitions)
        
        return {
            'track_id': track_id,
            'total_recognitions': len(recognitions),
            'recognition_counts': dict(recognition_counts),
            'most_common': most_common,
            'confidence': confidence,
            'duration': len(self.track_history[track_id])
        }
    
    def get_smoothed_bbox(self, track_id):
        """Get smoothed bounding box using Kalman filter prediction"""
        for track in self.tracks:
            if track['id'] == track_id:
                # Get predicted position
                predicted = track['kalman'].predict()
                center_x, center_y = predicted[0], predicted[1]
                
                # Get current bbox
                x, y, w, h = track['bbox']
                
                # Calculate smoothed bbox
                smoothed_x = center_x - w/2
                smoothed_y = center_y - h/2
                
                return (int(smoothed_x), int(smoothed_y), int(w), int(h))
        
        return None
