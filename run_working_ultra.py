#!/usr/bin/env python3
"""
Working Ultra Enhanced Face Recognition System - Main Runner
16 emotions with enhanced features
"""

import os
import sys

# Add directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'advanced'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'integrations'))

try:
    from working_ultra_system import main
    main()
except ImportError as e:
    print(f"Error importing working ultra system: {e}")
    print("Make sure integrations/working_ultra_system.py exists")
except Exception as e:
    print(f"Error running working ultra face recognition: {e}")
