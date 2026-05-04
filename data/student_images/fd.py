import pickle
import cv2
import face_recognition
import os
from imutils import paths # Ensure imutils is installed
from parameters import DLIB_FACE_ENCODING_PATH, DATASET_PATH

def create_face_embeddings():
    # Ensure the dataset directory exists
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset path {DATASET_PATH} does not exist.")
        return

    imagePaths = list(paths.list_images(DATASET_PATH))
    print(f"[INFO] Found {len(imagePaths)} images in {DATASET_PATH}")

    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):
        print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
        
        # Folder structure must be: DATASET_PATH/Person_Name/image.jpg
        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect the (x, y)-coordinates of the bounding boxes
        # Using 'hog' for speed, 'cnn' for accuracy (if GPU available)
        boxes = face_recognition.face_locations(rgb, model='hog')
        
        # Compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=10)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

    print(f"[INFO] Serializing {len(knownEncodings)} encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    
    os.makedirs(os.path.dirname(DLIB_FACE_ENCODING_PATH), exist_ok=True)
    with open(DLIB_FACE_ENCODING_PATH, "wb") as f:
        f.write(pickle.dumps(data))
    print("[INFO] Done.")

if __name__ == '__main__':
    create_face_embeddings()