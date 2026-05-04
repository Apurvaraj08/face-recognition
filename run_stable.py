#!/usr/bin/env python3
"""
Stable Face Recognition System - Main Runner
"""

import os
import sys

# Add core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

try:
    from stable_face_recognition import main
    main()
except ImportError as e:
    print(f"Error importing stable recognition system: {e}")
    print("Make sure core/stable_face_recognition.py exists")
except Exception as e:
    print(f"Error running stable face recognition: {e}")
