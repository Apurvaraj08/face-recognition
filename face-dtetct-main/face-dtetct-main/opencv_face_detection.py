"""
Simple OpenCV-only real-time face detector (uses Haar cascades).
Does not require dlib/face_recognition - suitable when dlib can't be built.
Controls:
  Q - Quit
  S - Save snapshot to reports/snapshots/

Run with the Python interpreter that has OpenCV installed (e.g. the venv at C:\pyenv):
C:\pyenv\Scripts\python.exe opencv_face_detection.py
"""

import cv2
import os
import time
from datetime import datetime

REPORTS_DIR = os.path.join(os.getcwd(), 'reports', 'snapshots')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load cascade
if hasattr(cv2, 'data') and cv2.data.haarcascades:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
else:
    # Fallback for environments where cv2.data is not populated (e.g. some Conda builds)
    import sys
    base_prefix = sys.base_prefix if hasattr(sys, 'base_prefix') else sys.prefix
    cascade_path = os.path.join(base_prefix, 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(cascade_path)

def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError('Could not open camera. Ensure a webcam is connected.')
    return cap


def process_frame(frame):
    """Process a single frame, drawing boxes and showing it."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Optimized parameters for real-time face detection
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,        # Standard scale
        minNeighbors=4,         # Balanced: not too strict, not too loose
        minSize=(30,30),        # Minimum face size
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Face{len(faces)}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display detection count
    cv2.putText(frame, f'Faces detected: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, 'Press: Q=Quit, S=Snapshot', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('OpenCV Face Detector', frame)


def main():
    cap = open_camera(0)
    
    # Verify cascade classifier loaded correctly
    if face_cascade.empty():
        print('[ERROR] Failed to load face cascade classifier!')
        return
    
    print('[INFO] OpenCV Face Detector - Ready')
    print('[INFO] Controls: Q=Quit, S=Snapshot')
    print('[INFO] Tips for better detection:')
    print('       * Ensure good lighting (preferably natural/bright light)')
    print('       * Face the camera directly (frontal face)')
    print('       * Keep face roughly 100-400 pixels wide on screen')
    print('       * Avoid sunglasses/masks that obscure features')
    print('[START] Detector is running...\n')

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('[WARNING] Failed to read frame from camera')
            time.sleep(0.1)
            continue

        # Mirror the frame for better user experience (selfie-like)
        frame = cv2.flip(frame, 1)
        
        # Process frame with face detection
        process_frame(frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print(f'\n[INFO] Exiting. Frames processed: {frame_count}')
            break
        if key == ord('s') or key == ord('S'):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = os.path.join(REPORTS_DIR, f'snapshot_{ts}.jpg')
            cv2.imwrite(fname, frame)
            print(f'[SNAPSHOT] Saved: {fname}')

    cap.release()
    cv2.destroyAllWindows()
    print('[INFO] Detector stopped.')

if __name__ == '__main__':
    main()
