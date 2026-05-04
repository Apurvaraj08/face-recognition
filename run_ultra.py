#!/usr/bin/env python3
"""
Ultra Enhanced Face Recognition System - Main Runner
"""

import os
import sys

# Add directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'advanced'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'integrations'))

try:
    from ultra_enhanced_system import main
    main()
except ImportError as e:
    print(f"Error importing ultra enhanced system: {e}")
    print("Make sure integrations/ultra_enhanced_system.py exists")
except Exception as e:
    print(f"Error running ultra enhanced face recognition: {e}")
