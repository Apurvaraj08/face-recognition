#!/usr/bin/env python3
"""
Batch Processing System for Face Recognition
Processes multiple images/videos in batch mode
"""

import os
import cv2
import numpy as np
import json
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict
import multiprocessing as mp

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports")
BATCH_DIR = os.path.join(ROOT, "batch")

class BatchProcessor:
    def __init__(self, recognition_system, max_workers=None):
        self.recognition_system = recognition_system
        self.max_workers = max_workers or mp.cpu_count()
        self.batch_results = []
        self.processing_stats = defaultdict(int)
        
        # Ensure batch directory exists
        os.makedirs(BATCH_DIR, exist_ok=True)
    
    def process_image_batch(self, image_paths, output_format='json'):
        """Process a batch of images"""
        print(f"🖼️  Processing {len(image_paths)} images in batch...")
        
        start_time = time.time()
        results = []
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for image_path in image_paths:
                future = executor.submit(self._process_single_image, image_path)
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                        self.processing_stats['successful'] += 1
                    else:
                        self.processing_stats['failed'] += 1
                except Exception as e:
                    print(f"❌ Error processing image: {e}")
                    self.processing_stats['failed'] += 1
        
        processing_time = time.time() - start_time
        
        # Save batch results
        batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._save_batch_results(results, batch_id, output_format)
        
        return {
            'batch_id': batch_id,
            'total_images': len(image_paths),
            'successful': self.processing_stats['successful'],
            'failed': self.processing_stats['failed'],
            'processing_time': processing_time,
            'results': results
        }
    
    def _process_single_image(self, image_path):
        """Process a single image"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not read image: {image_path}")
                return None
            
            # Detect faces
            faces = self.recognition_system.detect_faces_stable(image)
            
            # Process each face
            face_results = []
            for (x, y, w, h) in faces:
                face_region = image[y:y+h, x:x+w]
                
                # Recognize face
                name, similarity = self.recognition_system.recognize_face_stable(face_region)
                quality = self.recognition_system.assess_face_quality_stable(face_region)
                
                # Update attendance if recognized
                if name != "Unknown":
                    self.recognition_system.update_attendance(name)
                    self.recognition_system.recognition_stats[name] += 1
                
                face_results.append({
                    'location': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'recognition': {
                        'name': name,
                        'similarity': round(similarity, 3),
                        'quality': round(quality, 3)
                    }
                })
            
            return {
                'image_path': os.path.basename(image_path),
                'image_info': {
                    'dimensions': {'width': image.shape[1], 'height': image.shape[0]},
                    'faces_detected': len(faces)
                },
                'face_results': face_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None
    
    def process_video_batch(self, video_paths, frame_interval=30, output_format='json'):
        """Process a batch of videos"""
        print(f"🎥 Processing {len(video_paths)} videos in batch...")
        
        start_time = time.time()
        results = []
        
        for video_path in video_paths:
            try:
                video_result = self._process_single_video(video_path, frame_interval)
                if video_result:
                    results.append(video_result)
                    self.processing_stats['successful'] += 1
                else:
                    self.processing_stats['failed'] += 1
            except Exception as e:
                print(f"❌ Error processing video {video_path}: {e}")
                self.processing_stats['failed'] += 1
        
        processing_time = time.time() - start_time
        
        # Save batch results
        batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._save_batch_results(results, batch_id, output_format)
        
        return {
            'batch_id': batch_id,
            'total_videos': len(video_paths),
            'successful': self.processing_stats['successful'],
            'failed': self.processing_stats['failed'],
            'processing_time': processing_time,
            'frame_interval': frame_interval,
            'results': results
        }
    
    def _process_single_video(self, video_path, frame_interval):
        """Process a single video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Could not open video: {video_path}")
                return None
            
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"📹 Processing {os.path.basename(video_path)}: {total_frames} frames, {duration:.1f}s")
            
            frame_results = []
            frame_count = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every nth frame
                if frame_count % frame_interval == 0:
                    # Detect faces
                    faces = self.recognition_system.detect_faces_stable(frame)
                    
                    # Process faces
                    face_results = []
                    for (x, y, w, h) in faces:
                        face_region = frame[y:y+h, x:x+w]
                        
                        # Recognize face
                        name, similarity = self.recognition_system.recognize_face_stable(face_region)
                        quality = self.recognition_system.assess_face_quality_stable(face_region)
                        
                        # Update attendance if recognized
                        if name != "Unknown":
                            self.recognition_system.update_attendance(name)
                            self.recognition_system.recognition_stats[name] += 1
                        
                        face_results.append({
                            'location': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                            'recognition': {
                                'name': name,
                                'similarity': round(similarity, 3),
                                'quality': round(quality, 3)
                            }
                        })
                    
                    frame_results.append({
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'faces_detected': len(faces),
                        'face_results': face_results
                    })
                    
                    processed_frames += 1
                    
                    # Progress update
                    if processed_frames % 10 == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            cap.release()
            
            return {
                'video_path': os.path.basename(video_path),
                'video_info': {
                    'total_frames': total_frames,
                    'fps': fps,
                    'duration': duration,
                    'processed_frames': processed_frames,
                    'frame_interval': frame_interval
                },
                'frame_results': frame_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error processing video {video_path}: {e}")
            return None
    
    def process_directory(self, directory_path, recursive=True, file_types=None):
        """Process all supported files in a directory"""
        if file_types is None:
            file_types = ['.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov']
        
        print(f"📁 Scanning directory: {directory_path}")
        
        # Find all files
        image_files = []
        video_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_files.append(file_path)
                    elif file_ext in ['.mp4', '.avi', '.mov']:
                        video_files.append(file_path)
        else:
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_files.append(file_path)
                    elif file_ext in ['.mp4', '.avi', '.mov']:
                        video_files.append(file_path)
        
        print(f"📸 Found {len(image_files)} images")
        print(f"🎥 Found {len(video_files)} videos")
        
        results = {}
        
        # Process images
        if image_files:
            print("\n🖼️  Processing images...")
            image_results = self.process_image_batch(image_files)
            results['images'] = image_results
        
        # Process videos
        if video_files:
            print("\n🎥 Processing videos...")
            video_results = self.process_video_batch(video_files)
            results['videos'] = video_results
        
        return results
    
    def _save_batch_results(self, results, batch_id, output_format):
        """Save batch processing results"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        batch_data = {
            'batch_id': batch_id,
            'timestamp': timestamp,
            'processing_stats': dict(self.processing_stats),
            'results': results
        }
        
        if output_format == 'json':
            # Save as JSON
            json_path = os.path.join(BATCH_DIR, f"batch_results_{batch_id}.json")
            with open(json_path, 'w') as f:
                json.dump(batch_data, f, indent=2)
            print(f"💾 Batch results saved to: {json_path}")
        
        elif output_format == 'csv':
            # Save as CSV (flatten results)
            csv_path = os.path.join(BATCH_DIR, f"batch_results_{batch_id}.csv")
            import csv
            
            with open(csv_path, 'w', newline='') as csvfile:
                if results and results[0].get('image_path'):  # Image batch
                    fieldnames = ['batch_id', 'timestamp', 'image_path', 'faces_detected', 
                               'face_id', 'name', 'similarity', 'quality', 'x', 'y', 'width', 'height']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for result in results:
                        base_row = {
                            'batch_id': batch_id,
                            'timestamp': result['timestamp'],
                            'image_path': result['image_path'],
                            'faces_detected': len(result['face_results'])
                        }
                        
                        for i, face_result in enumerate(result['face_results']):
                            row = base_row.copy()
                            row.update({
                                'face_id': i + 1,
                                'name': face_result['recognition']['name'],
                                'similarity': face_result['recognition']['similarity'],
                                'quality': face_result['recognition']['quality'],
                                'x': face_result['location']['x'],
                                'y': face_result['location']['y'],
                                'width': face_result['location']['width'],
                                'height': face_result['location']['height']
                            })
                            writer.writerow(row)
                        
                        if not result['face_results']:  # No faces detected
                            writer.writerow(base_row)
            
            print(f"💾 Batch results saved to: {csv_path}")
        
        return batch_data
    
    def generate_batch_report(self, batch_id=None):
        """Generate comprehensive batch processing report"""
        if batch_id:
            # Generate report for specific batch
            json_path = os.path.join(BATCH_DIR, f"batch_results_{batch_id}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    batch_data = json.load(f)
                return self._create_batch_report(batch_data)
            else:
                print(f"❌ Batch {batch_id} not found")
                return None
        else:
            # Generate summary report for all batches
            all_reports = []
            
            for file in os.listdir(BATCH_DIR):
                if file.startswith('batch_results_') and file.endswith('.json'):
                    file_path = os.path.join(BATCH_DIR, file)
                    try:
                        with open(file_path, 'r') as f:
                            batch_data = json.load(f)
                        report = self._create_batch_report(batch_data)
                        if report:
                            all_reports.append(report)
                    except:
                        continue
            
            summary_report = {
                'report_type': 'batch_summary',
                'timestamp': datetime.now().isoformat(),
                'total_batches': len(all_reports),
                'batch_reports': all_reports
            }
            
            # Save summary report
            summary_path = os.path.join(REPORTS_DIR, f"batch_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
            with open(summary_path, 'w') as f:
                json.dump(summary_report, f, indent=2)
            
            print(f"📊 Batch summary report saved to: {summary_path}")
            return summary_report
    
    def _create_batch_report(self, batch_data):
        """Create detailed report for batch data"""
        timestamp = batch_data['timestamp']
        batch_id = batch_data['batch_id']
        processing_stats = batch_data['processing_stats']
        results = batch_data['results']
        
        # Analyze results
        total_faces = 0
        recognized_faces = 0
        student_counts = defaultdict(int)
        quality_scores = []
        
        if isinstance(results, list):  # Single batch
            for result in results:
                if 'face_results' in result:
                    total_faces += len(result['face_results'])
                    for face_result in result['face_results']:
                        if face_result['recognition']['name'] != 'Unknown':
                            recognized_faces += 1
                            student_counts[face_result['recognition']['name']] += 1
                        quality_scores.append(face_result['recognition']['quality'])
        
        # Calculate statistics
        recognition_rate = (recognized_faces / total_faces * 100) if total_faces > 0 else 0
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        report = {
            'batch_id': batch_id,
            'timestamp': timestamp,
            'processing_stats': processing_stats,
            'analysis': {
                'total_faces': total_faces,
                'recognized_faces': recognized_faces,
                'recognition_rate': round(recognition_rate, 2),
                'avg_quality': round(avg_quality, 3),
                'student_counts': dict(student_counts)
            }
        }
        
        # Save detailed report
        report_path = os.path.join(REPORTS_DIR, f"batch_report_{batch_id}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Batch report saved to: {report_path}")
        return report

# Batch processing scheduler
class BatchScheduler:
    def __init__(self, recognition_system):
        self.recognition_system = recognition_system
        self.processor = BatchProcessor(recognition_system)
        self.scheduled_jobs = []
    
    def schedule_directory_processing(self, directory_path, schedule_time=None, recursive=True):
        """Schedule directory processing"""
        job = {
            'type': 'directory',
            'path': directory_path,
            'schedule_time': schedule_time,
            'recursive': recursive,
            'status': 'scheduled',
            'created_at': datetime.now()
        }
        
        self.scheduled_jobs.append(job)
        
        if schedule_time is None:
            # Process immediately
            self._process_job(job)
        
        return job
    
    def _process_job(self, job):
        """Process a scheduled job"""
        job['status'] = 'processing'
        job['started_at'] = datetime.now()
        
        try:
            if job['type'] == 'directory':
                results = self.processor.process_directory(
                    job['path'], 
                    recursive=job.get('recursive', True)
                )
                job['results'] = results
                job['status'] = 'completed'
            else:
                job['status'] = 'failed'
                job['error'] = 'Unknown job type'
        
        except Exception as e:
            job['status'] = 'failed'
            job['error'] = str(e)
        
        job['completed_at'] = datetime.now()
        
        # Generate report
        if job['status'] == 'completed':
            self.processor.generate_batch_report()
    
    def get_job_status(self, job_id):
        """Get status of a specific job"""
        for job in self.scheduled_jobs:
            if id(job) == job_id:
                return job
        return None
    
    def get_all_jobs(self):
        """Get all scheduled jobs"""
        return self.scheduled_jobs

# Example usage
def run_batch_processing_example():
    """Example of batch processing usage"""
    print("🚀 Batch Processing Example")
    print("This would integrate with the main recognition system")
    print("Usage:")
    print("1. processor = BatchProcessor(recognition_system)")
    print("2. results = processor.process_directory('/path/to/images')")
    print("3. processor.generate_batch_report()")

if __name__ == "__main__":
    run_batch_processing_example()
