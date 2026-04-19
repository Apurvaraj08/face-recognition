'''
Real-time face recognition using dlib + face_recognition library.
Uses pre-computed embeddings from dlib_face_embeddings.py.

Controls:
  Q - Quit
  S - Save current frame snapshot to reports/
  R - Toggle recording on/off
'''

import cv2
import face_recognition
import pickle
import os
import time
import numpy as np
from datetime import datetime

from parameters import (
    DLIB_FACE_ENCODING_PATH,
    FACE_MATCHING_TOLERANCE,
    FACE_RECOGNITION_MODEL,
    NUMBER_OF_TIMES_TO_UPSAMPLE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    REPORT_PATH,
)

# -- Colours and font ----------------------------------------------------------
KNOWN_COLOUR   = (0, 200, 0)    # green  - recognised face
UNKNOWN_COLOUR = (0, 0, 220)    # red    - unknown face
TEXT_COLOUR    = (255, 255, 255)
FONT           = cv2.FONT_HERSHEY_SIMPLEX

# -- Process every N-th frame for speed (1 = every frame) ---------------------
PROCESS_EVERY_N_FRAMES = 2


def load_encodings(path: str):
    '''Load pre-computed face encodings from a pickle file.
    The incoming path may be relative; if so it is interpreted relative to
    this script's directory so you can run from anywhere in the repo.'''  
    # resolve relative path to script directory
    if not os.path.isabs(path):
        base = os.path.dirname(__file__)
        path = os.path.join(base, path)
    print(f'[INFO] Loading encodings from {path} ...')
    with open(path, 'rb') as f:
        data = pickle.loads(f.read())
    print(f'[INFO] Loaded {len(data["names"])} face encoding(s).')
    return data['encodings'], data['names']


def open_camera():
    '''Try to open the default webcam (index 0) with proper initialization.'''
    print('[INFO] Attempting to open camera...')
    
    # Try with DSHOW backend first (more stable on Windows), fallback to default
    backends = [
        (cv2.CAP_DSHOW, 'DSHOW'),
        (cv2.CAP_MSMF, 'MSMF'),
        (-1, 'DEFAULT')
    ]
    
    cap = None
    for backend, name in backends:
        try:
            if backend == -1:
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(0, backend)
            
            if cap.isOpened():
                print(f'[INFO] Camera opened with {name} backend')
                break
            else:
                print(f'[WARN] {name} backend failed, trying next...')
                cap.release()
                cap = None
        except Exception as e:
            print(f'[WARN] {name} backend error: {e}')
            if cap:
                cap.release()
            cap = None
    
    if cap is None or not cap.isOpened():
        raise RuntimeError('Could not open camera. '
                           'Make sure a webcam is connected and drivers are installed.')
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get fresh frames
    
    # Give camera time to initialize and warm up
    print('[INFO] Warming up camera (waiting 2 seconds)...')
    time.sleep(2)
    
    # Try to grab and discard a few frames to ensure camera is ready
    for i in range(5):
        ret, _ = cap.read()
        if ret:
            print(f'[INFO] Camera ready after {i+1} frame(s)')
            break
        time.sleep(0.2)
    
    print(f'[INFO] Camera initialized at {FRAME_WIDTH}x{FRAME_HEIGHT}')
    return cap


def draw_box(frame, top, right, bottom, left, name, colour):
    '''Draw a labelled bounding box around a detected face.'''
    cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
    cv2.rectangle(frame, (left, bottom - 28), (right, bottom), colour, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 8),
                FONT, 0.65, TEXT_COLOUR, 1)


def run_recognition():
    # -- sanity check: numpy/dlib compatibility ---------------------------------
    try:
        import numpy as _np, dlib as _dlib
        if tuple(int(x) for x in _np.__version__.split('.')[:1]) >= (2,):
            print(f'[WARN] Detected numpy {_np.__version__}; dlib may not support numpy>=2.')
            print('       Consider using a conda-forge environment with numpy 1.x.')
    except ImportError:
        pass

    # -- Load embeddings -------------------------------------------------------
    try:
        known_encodings, known_names = load_encodings(DLIB_FACE_ENCODING_PATH)
    except (FileNotFoundError, pickle.UnpicklingError) as exc:
        # fall back to the OpenCV-only detector if the file is missing or corrupt
        print(f'[WARN] Unable to load encodings ({exc}); switching to simple OpenCV detector.')
        import opencv_face_detection as cvdet
        cap = open_camera()
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            cvdet.process_frame(frame)  # helper function, see import below
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    # -- Open camera -----------------------------------------------------------
    cap = open_camera()

    # -- Prepare output directories --------------------------------------------
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    snapshots_dir = os.path.join(os.path.dirname(REPORT_PATH), 'snapshots')
    os.makedirs(snapshots_dir, exist_ok=True)

    # -- Video writer (optional recording) -------------------------------------
    recording   = False
    video_writer = None

    frame_count    = 0
    face_locations = []
    face_names     = []

    print('[INFO] Starting face recognition. Press Q to quit, S to snapshot, R to record.')

    # -- Recognition CSV log ---------------------------------------------------
    seen_names = set()
    consecutive_failures = 0

    consecutive_failures = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures <= 3:
                print(f'[WARN] Failed to grab frame (attempt {consecutive_failures}). Retrying...')
                time.sleep(0.2)  # Slightly longer wait
            elif consecutive_failures <= 10:
                print(f'[WARN] Persistent frame grab failure (attempt {consecutive_failures}). Waiting longer...')
                time.sleep(0.5)
            else:
                print('[ERROR] Camera appears to be disconnected or unresponsive. Exiting...')
                break
            continue
        
        consecutive_failures = 0  # Reset counter on successful frame grab

        frame_count += 1
        display = frame.copy()

        # -- Only process every N-th frame -------------------------------------
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # Shrink frame to 1/4 size for faster face detection
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # sanity check: OpenCV occasionally returns unexpected dtypes/empty frames
            if small is None or small.size == 0:
                print('[WARN] Received empty frame when resizing, skipping this iteration')
                continue
            if small.dtype != np.uint8:
                # convert/cast to uint8 so face_recognition will accept it
                small = small.astype(np.uint8)

            try:
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f'[ERROR] cvtColor failed: {e}, frame dtype={small.dtype}, shape={small.shape}')
                # skip detection rather than crashing
                continue

            try:
                face_locations = face_recognition.face_locations(
                    rgb_small,
                    number_of_times_to_upsample=NUMBER_OF_TIMES_TO_UPSAMPLE,
                    model=FACE_RECOGNITION_MODEL,
                )
                face_encodings = face_recognition.face_encodings(
                    rgb_small,
                    known_face_locations=face_locations,
                    num_jitters=1,
                    model='small',
                )
            except Exception as err:
                # capture the problematic input for debugging
                print('[ERROR] face_recognition failure:', err)
                print(f'       rgb_small dtype={rgb_small.dtype}, shape={rgb_small.shape}')
                # continue looping rather than aborting
                face_locations = []
                face_encodings = []

            face_names = []
            for encoding in face_encodings:
                distances = face_recognition.face_distance(known_encodings, encoding)
                best_idx  = int(np.argmin(distances)) if len(distances) else -1
                name      = 'Unknown'
                if best_idx >= 0 and distances[best_idx] <= FACE_MATCHING_TOLERANCE:
                    name = known_names[best_idx]
                    seen_names.add(name)
                face_names.append(name)

        # -- Draw boxes (scale coords back up x4) ------------------------------
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top    *= 4;  right  *= 4
            bottom *= 4;  left   *= 4
            colour  = KNOWN_COLOUR if name != 'Unknown' else UNKNOWN_COLOUR
            draw_box(display, top, right, bottom, left, name, colour)

        # -- HUD overlay -------------------------------------------------------
        ts  = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        fps_text = f'Frame: {frame_count}  |  {ts}'
        rec_text = 'REC' if recording else ''
        cv2.putText(display, fps_text, (10, 25), FONT, 0.55, (200, 200, 200), 1)
        if rec_text:
            cv2.putText(display, rec_text, (FRAME_WIDTH - 80, 25),
                        FONT, 0.65, (0, 0, 255), 2)

        # -- Optional recording -------------------------------------------------
        if recording and video_writer:
            video_writer.write(display)

        cv2.imshow('GRIL Team Face Recognition  [Q=quit  S=snapshot  R=record]', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:   # Q or ESC -> quit
            break

        elif key == ord('s'):              # S -> save snapshot
            snap_path = os.path.join(
                snapshots_dir,
                f'snapshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            )
            cv2.imwrite(snap_path, display)
            print(f'[INFO] Snapshot saved -> {snap_path}')

        elif key == ord('r'):              # R -> toggle recording
            recording = not recording
            if recording:
                vid_path = os.path.join(
                    snapshots_dir,
                    f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi'
                )
                fourcc       = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(vid_path, fourcc, 20.0,
                                               (FRAME_WIDTH, FRAME_HEIGHT))
                print(f'[INFO] Recording started -> {vid_path}')
            else:
                if video_writer:
                    video_writer.release()
                    video_writer = None
                print('[INFO] Recording stopped.')

    # -- Cleanup ---------------------------------------------------------------
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    # -- Save attendance-style CSV ----------------------------------------------
    if seen_names:
        ts_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_lines = [f'name,seen_at\n']
        for n in sorted(seen_names):
            report_lines.append(f'{n},{ts_str}\n')
        with open(REPORT_PATH, 'w') as f:
            f.writelines(report_lines)
        print(f'[INFO] Attendance report saved -> {REPORT_PATH}')

    print(f'[INFO] Recognised faces: {sorted(seen_names) if seen_names else "none"}')
    print('[INFO] Done.')


if __name__ == '__main__':
    run_recognition()
