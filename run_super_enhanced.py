#!/usr/bin/env python3
"""
Super Enhanced Face Recognition System - Main Runner
25+ emotions with enhanced face detection
"""

import os
import sys

# Add directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'advanced'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'integrations'))

try:
    from super_enhanced_system import main
    main()
except ImportError as e:
    print(f"Error importing super enhanced system: {e}")
    print("Make sure integrations/super_enhanced_system.py exists")
except Exception as e:
    print(f"Error running super enhanced face recognition: {e}")
