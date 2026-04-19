"""
Simple camera test - checks if camera is accessible
"""
import cv2
import sys

print('[TEST] Attempting to open camera with different methods...\n')

# Try different camera backends
backends = [
    (cv2.CAP_DSHOW, 'DSHOW (DirectShow - Windows)'),
    (cv2.CAP_ANY, 'CAP_ANY (Auto-detect)'),
    (0, 'Index 0 (Default)'),
]

camera_found = False

for backend_id, backend_name in backends:
    try:
        if isinstance(backend_id, str):
            continue
            
        print(f'[{backend_name}] Attempting to open camera...')
        cap = cv2.VideoCapture(backend_id)
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if ret and frame is not None:
            print(f'  [OK] SUCCESS! Camera opened with {backend_name}')
            print(f'    Frame size: {frame.shape[1]}x{frame.shape[0]}')
            
            # Try grabbing a few frames to test stability
            for i in range(5):
                ret, frame = cap.read()
                if not ret:
                    print(f'    [FAIL] Failed to read frame {i+1}')
                    break
            else:
                print(f'    [OK] Successfully read 5 frames')
                camera_found = True
        else:
            print(f'  [FAIL] Failed - could not read frame')
        
        cap.release()
        print()
        
        if camera_found:
            break
            
    except Exception as e:
        print(f'  [FAIL] Error: {e}\n')

if camera_found:
    print('[SUCCESS] Camera is working and accessible!')
    sys.exit(0)
else:
    print('[ERROR] Camera could not be opened. Possible causes:')
    print('  1. No camera connected to computer')
    print('  2. Camera in use by another application')
    print('  3. Camera drivers not installed')
    print('  4. Camera disabled in BIOS/Windows settings')
    print('\nTo fix:')
    print('  - Check Device Manager (Cameras section)')
    print('  - Close other apps using camera (Zoom, Skype, etc.)')
    print('  - Update camera drivers')
    sys.exit(1)
