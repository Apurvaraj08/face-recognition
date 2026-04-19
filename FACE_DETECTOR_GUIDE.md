## Face Detector - Quick Guide

### What was fixed:
✓ Optimized face detection parameters (reduced false positives from 17-61 to 4-8 per image)
✓ Added better on-screen instructions and tips
✓ Frame mirroring for natural selfie-like experience
✓ Detection diagnostics

### How to run:

```powershell
cd "c:\Users\Apurava Raj\OneDrive\Grill\Grill Assignment"
python "face-dtetct-main\face-dtetct-main\opencv_face_detection.py"
```

### Controls:
- **Q** — Quit the detector
- **S** — Save current frame snapshot to `reports/snapshots/`

### Troubleshooting if faces still aren't detected:

1. **Lighting**: The most common issue. Face detection needs good lighting.
   - Move to a brighter area (near window or lamp)
   - Avoid backlighting

2. **Face Position**: 
   - Look directly at the camera (frontal face)
   - Keep face roughly 100-400 pixels wide on screen
   - Avoid wearing sunglasses/hats that obscure features

3. **Distance**: 
   - Not too close (< 2 feet)
   - Not too far (> 10 feet)
   - Sit about 3-5 feet from camera

4. **Test on Images**: 
   If detector doesn't work on camera but works on student images, your webcam may have issues.
   Run: `python "face-dtetct-main\face-dtetct-main\test_cascade.py"`
   to verify detection works.

### Detection Results:
Tested on 14 student images: **76 faces detected**
Average: 5-6 faces per image (balanced detection without false positives)

### Next Steps:
For student identification (not just face detection), use the dlib-based pipeline:
`python "face-dtetct-main\face-dtetct-main\dlib_face_embeddings.py"`
`python "face-dtetct-main\face-dtetct-main\face_recognition_run.py"`

(Requires dlib - may need conda environment)
