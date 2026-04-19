import os
import sys
import subprocess

ROOT = os.path.abspath(os.path.dirname(__file__))
STUDENT_IMAGES = os.path.abspath(r"C:\Users\Apurava Raj\OneDrive\Grill\Grill Assignment\Student Images")
EMBEDDINGS_SCRIPT = os.path.join(ROOT, "face-dtetct-main", "dlib_face_embeddings.py")
RECOGNITION_SCRIPT = os.path.join(ROOT, "face-dtetct-main", "face_recognition_run.py")

REQUIRED_MODULES = ["cv2", "dlib", "face_recognition", "numpy", "imutils"]


def check_python_modules():
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
        except Exception as e:
            missing.append((mod, str(e)))
    return missing


def install_missing_modules(modules):
    if not modules:
        return True

    to_install = ["opencv-python", "dlib", "face_recognition", "imutils", "numpy"]
    print("Missing required modules detected:")
    for mod in modules:
        print(f" - {mod[0]}")

    answer = input("Install missing modules now via pip? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        return False

    print("Installing dependencies...")
    cmd = [sys.executable, "-m", "pip", "install"] + to_install
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False, text=True)
    return result.returncode == 0


def run_script(script_path, args=None):
    args = args or []
    cmd = [sys.executable, script_path] + args
    print(f"Running: {cmd}")
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Script failed: {script_path} (exit {completed.returncode})")


def main():
    print("=== Grill Assignment Full Pipeline Runner ===")
    print(f"Using Python: {sys.executable}")
    if not os.path.isdir(STUDENT_IMAGES):
        raise FileNotFoundError(f"Student images folder not found: {STUDENT_IMAGES}")

    print("Checking dependencies...")
    missing = check_python_modules()
    if missing:
        print("Error: missing required Python modules")
        for mod, err in missing:
            print(f" - {mod}: {err}")
        if not install_missing_modules(missing):
            print("Dependency installation skipped or failed. Exiting.")
            sys.exit(1)
        print("Re-checking dependencies...")
        missing = check_python_modules()
        if missing:
            print("Dependencies are still missing after install. Please fix manually and retry.")
            sys.exit(1)

    if not os.path.isfile(EMBEDDINGS_SCRIPT):
        raise FileNotFoundError(f"json path not found: {EMBEDDINGS_SCRIPT}")
    if not os.path.isfile(RECOGNITION_SCRIPT):
        raise FileNotFoundError(f"json path not found: {RECOGNITION_SCRIPT}")

    print("Generating embeddings...")
    run_script(EMBEDDINGS_SCRIPT, [STUDENT_IMAGES])
    print("Starting recognition loop...")
    run_script(RECOGNITION_SCRIPT)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print("\nERROR:", ex)
        print("Please check dependencies and paths, then retry.")
        sys.exit(2)
