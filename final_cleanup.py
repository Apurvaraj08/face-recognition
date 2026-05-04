#!/usr/bin/env python3
"""
Final Project Cleanup Script
Remove unnecessary files and organize the project structure
"""

import os
import shutil
import glob
from datetime import datetime

ROOT = os.path.abspath(os.path.dirname(__file__))

class FinalProjectCleaner:
    def __init__(self):
        # Essential files to keep
        self.essential_files = set([
            'src/stable_face_recognition.py',
            'src/super_enhanced_system.py',
            'src/expanded_emotion_detection.py',
            'src/enhanced_face_detection.py',
            'src/working_ultra_system.py',
            'src/ultra_enhanced_system.py',
            'src/advanced_emotion_detection.py',
            'src/advanced_face_detection.py',
            'src/face_tracking.py',
            'src/alert_system.py',
            'src/analytics_dashboard.py',
            'src/multi_camera_system.py',
            'src/face_recognition_api.py',
            'src/batch_processor.py',
            'src/automated_reports.py',
            'data/face_encodings.pkl',
            'data/student_images',
            'config/settings.json',
            'reports',
            'alerts',
            'analytics',
            'automated_reports',
            'batch',
            'uploads',
            'emotion_data',
            'run_stable.py',
            'run_working_ultra.py',
            'run_super_enhanced.py',
            'run_ultra.py'
        ])
        
        # Unnecessary file patterns to remove
        self.unnecessary_patterns = [
            'enhanced_*.py',
            'emotion_detection.py',  # Replaced by expanded_emotion_detection.py
            'face_recognition_main.py',
            'recognition_diagnostic.py',
            'project_*.py',
            'cleanup_*.py',
            'compressor_*.py',
            'demonstration_*.py',
            'test_*.py',
            'diagnostic_*.py',
            'FINAL_*.md',
            'PROJECT_*.md',
            '*.pyc',
            '__pycache__',
            '*.log',
            '*.tmp',
            'stable_detection_*.jpg',
            'enhanced_detection_*.jpg',
            'face_detection_*.jpg',
            'reference_*.jpg',
            'processed_*.jpg',
            'cleanup_project*.py',
            'cleanup_project_fixed*.py'
        ]
    
    def clean_project(self):
        """Clean up unnecessary files"""
        print("=== Final Project Cleanup ===")
        
        removed_count = 0
        removed_size = 0
        
        # Remove files matching patterns
        for pattern in self.unnecessary_patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    if os.path.isfile(file):
                        size = os.path.getsize(file)
                        os.remove(file)
                        removed_count += 1
                        removed_size += size
                        print(f"🗑️  Removed: {file}")
                    elif os.path.isdir(file):
                        size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                                 for dirpath, dirnames, filenames in os.walk(file) 
                                 for filename in filenames)
                        shutil.rmtree(file)
                        removed_count += 1
                        removed_size += size
                        print(f"🗑️  Removed directory: {file}")
                except Exception as e:
                    print(f"❌ Error removing {file}: {e}")
        
        # Remove specific unnecessary files
        specific_files = [
            'run_enhanced.py',
            'enhanced_main_system.py',
            'face-dtetct-main',
            'FACE_DETECTOR_GUIDE.md'
        ]
        
        for file in specific_files:
            if os.path.exists(file):
                try:
                    if os.path.isfile(file):
                        size = os.path.getsize(file)
                        os.remove(file)
                        removed_count += 1
                        removed_size += size
                        print(f"🗑️  Removed: {file}")
                    elif os.path.isdir(file):
                        size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                                 for dirpath, dirnames, filenames in os.walk(file) 
                                 for filename in filenames)
                        shutil.rmtree(file)
                        removed_count += 1
                        removed_size += size
                        print(f"🗑️  Removed directory: {file}")
                except Exception as e:
                    print(f"❌ Error removing {file}: {e}")
        
        print(f"✅ Total files removed: {removed_count}")
        print(f"💾 Space saved: {removed_size / 1024:.1f} KB")
        
        return removed_count, removed_size
    
    def organize_project_structure(self):
        """Organize project structure"""
        print("\n=== Organizing Final Project Structure ===")
        
        # Create organized directory structure
        organized_dirs = [
            'core',
            'advanced',
            'integrations',
            'docs'
        ]
        
        for dir_name in organized_dirs:
            dir_path = os.path.join(ROOT, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 Created directory: {dir_path}")
        
        # Move files to appropriate directories
        file_moves = {
            'core': ['stable_face_recognition.py'],
            'advanced': [
                'expanded_emotion_detection.py',
                'enhanced_face_detection.py',
                'face_tracking.py',
                'advanced_emotion_detection.py',
                'advanced_face_detection.py'
            ],
            'integrations': [
                'alert_system.py',
                'analytics_dashboard.py',
                'multi_camera_system.py',
                'face_recognition_api.py',
                'batch_processor.py',
                'automated_reports.py',
                'super_enhanced_system.py',
                'working_ultra_system.py',
                'ultra_enhanced_system.py'
            ],
            'docs': []
        }
        
        src_dir = os.path.join(ROOT, 'src')
        for category, files in file_moves.items():
            category_dir = os.path.join(ROOT, category)
            for file in files:
                src_file = os.path.join(src_dir, file)
                if os.path.exists(src_file):
                    dst_file = os.path.join(category_dir, file)
                    try:
                        shutil.move(src_file, dst_file)
                        print(f"📋 Moved: {file} -> {category}/")
                    except Exception as e:
                        print(f"❌ Error moving {file}: {e}")
        
        # Update run scripts to use new structure
        self._update_run_scripts()
        
        # Remove empty src directory if it's empty
        if os.path.exists(src_dir) and not os.listdir(src_dir):
            try:
                os.rmdir(src_dir)
                print(f"🗑️  Removed empty src directory")
            except:
                pass
    
    def _update_run_scripts(self):
        """Update run scripts for new structure"""
        print("\n=== Updating Run Scripts ===")
        
        # Update run_stable.py
        run_stable_path = os.path.join(ROOT, 'run_stable.py')
        if os.path.exists(run_stable_path):
            new_content = '''#!/usr/bin/env python3
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
'''
            with open(run_stable_path, 'w') as f:
                f.write(new_content)
            print("✅ Updated run_stable.py")
        
        # Update run_super_enhanced.py
        run_super_path = os.path.join(ROOT, 'run_super_enhanced.py')
        if os.path.exists(run_super_path):
            new_content = '''#!/usr/bin/env python3
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
'''
            with open(run_super_path, 'w') as f:
                f.write(new_content)
            print("✅ Updated run_super_enhanced.py")
        
        # Update run_working_ultra.py
        run_working_path = os.path.join(ROOT, 'run_working_ultra.py')
        if os.path.exists(run_working_path):
            new_content = '''#!/usr/bin/env python3
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
'''
            with open(run_working_path, 'w') as f:
                f.write(new_content)
            print("✅ Updated run_working_ultra.py")
        
        # Update run_ultra.py
        run_ultra_path = os.path.join(ROOT, 'run_ultra.py')
        if os.path.exists(run_ultra_path):
            new_content = '''#!/usr/bin/env python3
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
'''
            with open(run_ultra_path, 'w') as f:
                f.write(new_content)
            print("✅ Updated run_ultra.py")
    
    def generate_final_summary(self):
        """Generate final project summary"""
        print("\n=== Generating Final Project Summary ===")
        
        summary = {
            'project_name': 'Super Enhanced Face Recognition System',
            'version': '3.0',
            'timestamp': datetime.now().isoformat(),
            'structure': {
                'core/': 'Core recognition system',
                'advanced/': 'Advanced detection and emotion analysis (25+ emotions)',
                'integrations/': 'System integrations and features',
                'data/': 'Reference data and encodings',
                'config/': 'Configuration files',
                'reports/': 'Generated reports',
                'docs/': 'Documentation'
            },
            'features': [
                '31 Emotion Detection States (expanded from 16)',
                'Enhanced Face Detection (4 cascades + advanced preprocessing)',
                'Advanced Facial Expression Analysis',
                'Multiple Algorithm Detection',
                'Real-time Alerts & Landmark Detection',
                'Quality Assessment (10 metrics)',
                'Face Tracking with Kalman Filter',
                'Age & Gender Estimation',
                'Face Pose Estimation',
                'Analytics Dashboard',
                'Multi-Camera Support',
                'REST API',
                'Batch Processing',
                'Automated Reports'
            ],
            'run_scripts': {
                'run_stable.py': 'Run stable recognition system',
                'run_working_ultra.py': 'Run working ultra system (16 emotions)',
                'run_ultra.py': 'Run ultra enhanced system',
                'run_super_enhanced.py': 'Run super enhanced system (31 emotions)'
            },
            'system_hierarchy': {
                'stable': 'Basic stable recognition',
                'working_ultra': '16 emotions + enhanced detection',
                'ultra_enhanced': 'Advanced features (may have compatibility issues)',
                'super_enhanced': '31 emotions + best detection (recommended)'
            }
        }
        
        # Save summary
        summary_path = os.path.join(ROOT, 'docs', 'FINAL_PROJECT_SUMMARY.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        print(f"📋 Final project summary saved to: {summary_path}")
        
        # Create README
        readme_path = os.path.join(ROOT, 'README.md')
        readme_content = f"""# {summary['project_name']}

## Version {summary['version']}

## Features
{chr(10).join(f"- {feature}" for feature in summary['features'])}

## Project Structure
{chr(10).join(f"- **{dir}**: {desc}" for dir, desc in summary['structure'].items())}

## Usage
{chr(10).join(f"- `{script}`: {desc}" for script, desc in summary['run_scripts'].items())}

## System Hierarchy
{chr(10).join(f"- **{system}**: {desc}" for system, desc in summary['system_hierarchy'].items())}

## Recommended Usage
For best performance and features, use:
```bash
python run_super_enhanced.py --mode super_enhanced
```

This provides 31 emotion detection states with the most advanced face detection and preprocessing.
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"📄 README.md created")
        
        # Display summary
        print(f"\n📊 Final Project Summary:")
        print(f"Name: {summary['project_name']}")
        print(f"Version: {summary['version']}")
        print(f"Features: {len(summary['features'])} advanced features")
        print(f"Emotions: 31 detection states")
        print(f"Structure: {len(summary['structure'])} organized directories")
        print(f"Run Scripts: {len(summary['run_scripts'])} available")
        print(f"Recommended: Super Enhanced System (31 emotions)")

def main():
    """Main cleanup function"""
    cleaner = FinalProjectCleaner()
    
    print("🧹 Starting Final Project Cleanup and Organization")
    print("=" * 60)
    
    # Clean up files
    cleaner.clean_project()
    
    # Organize structure
    cleaner.organize_project_structure()
    
    # Generate summary
    cleaner.generate_final_summary()
    
    print("\n🎉 Final project cleanup and organization completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
