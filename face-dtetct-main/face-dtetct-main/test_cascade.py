"""
Test script to verify face cascade classifier and detect faces in Student Images folder.
"""

import cv2
import os
from pathlib import Path

# Load cascade
if hasattr(cv2, 'data') and cv2.data.haarcascades:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
else:
    # Fallback for environments where cv2.data is not populated (e.g. some Conda builds)
    import sys
    base_prefix = sys.base_prefix if hasattr(sys, 'base_prefix') else sys.prefix
    cascade_path = os.path.join(base_prefix, 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print('[ERROR] Failed to load face cascade!')
    exit(1)

print('[INFO] Cascade classifier loaded successfully')

# Test on student images
student_images_dir = r'C:\Users\Apurava Raj\OneDrive\Grill\Grill Assignment\Student Images'

supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
image_count = 0
faces_found = 0

for root, dirs, files in os.walk(student_images_dir):
    student_name = os.path.basename(root)
    if student_name == 'Student Images':
        student_name = 'Root'
    
    for file in files:
        if file.lower().endswith(supported_formats):
            img_path = os.path.join(root, file)
            image_count += 1
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f'[SKIP] Could not read: {img_path}')
                continue
            
            # Detect faces with optimized parameters
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30,30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                faces_found += len(faces)
                print(f'[OK] [{student_name}] {file}: {len(faces)} face(s) detected')
            else:
                print(f'[FAIL] [{student_name}] {file}: No faces detected')

print(f'\n[SUMMARY] Processed {image_count} images, found {faces_found} total faces')
