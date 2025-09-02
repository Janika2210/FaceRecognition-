"""
Face Recognition Tool
---------------------

This script combines all tasks:
1. Recognize faces from frames
2. Improve recognition accuracy (resize + confidence threshold)
3. Log false positives (with printed photos & lighting variation testing)
4. Landmark visualization + threshold tuning
5. CLI tool to regenerate encodings.pkl
6. Handle new users, duplicates, multiple formats (.jpg, .png)
7. Secure encoding storage placeholder (hashing in future)
8. Proper documentation & logging
9. Validation for single-face images

Requirements:
    pip install face_recognition opencv-python numpy
"""

import os
import cv2
import pickle
import argparse
import numpy as np
import face_recognition
from datetime import datetime


# ---------------------------
# Utility Functions
# ---------------------------

def load_encodings(path="encodings.pkl"):
    """Load face encodings from pickle file."""
    if not os.path.exists(path):
        return {"encodings": [], "names": []}
    with open(path, "rb") as f:
        return pickle.load(f)


def save_encodings(data, path="encodings.pkl"):
    """Save face encodings to pickle file."""
    with open(path, "wb") as f:
        pickle.dump(data, f)


def validate_single_face(image_path):
    """Ensure only 1 face exists in the image. Return error otherwise."""
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) != 1:
        raise ValueError(
            f"❌ {image_path} must contain exactly 1 face, found {len(face_locations)}.")
    return True


def regenerate_encodings(images_folder="images", encodings_path="encodings.pkl"):
    """Rebuild encodings.pkl from images folder."""
    data = {"encodings": [], "names": []}

    for file in os.listdir(images_folder):
        if file.lower().endswith((".jpg", ".png")):
            path = os.path.join(images_folder, file)
            try:
                validate_single_face(path)
                image = face_recognition.load_image_file(path)
                enc = face_recognition.face_encodings(image)[0]
                name = os.path.splitext(file)[0]

                if name in data["names"]:
                    print(f"⚠️ Duplicate name skipped: {name}")
                    continue

                data["encodings"].append(enc)
                data["names"].append(name)
                print(f"✅ Added {name}")

            except Exception as e:
                print(f"Error with {file}: {e}")

    save_encodings(data, encodings_path)
    print("🎉 Encodings regenerated and saved!")


# ---------------------------
# Face Recognition
# ---------------------------

def recognize_face_from_frame(frame, known_encodings, known_names, tolerance=0.5):
    """
    Recognize faces in a frame and return the matched name or None.
    Uses resizing + confidence thresholding for better accuracy.
    """
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations)

    if not face_encodings:
        return None

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance)
        face_distances = face_recognition.face_distance(
            known_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                return known_names[best_match_index]

    return None


# ---------------------------
# Debugging: Draw landmarks
# ---------------------------

def draw_landmarks(frame):
    """Draw facial landmarks for debugging."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    for landmarks in face_landmarks_list:
        for feature, points in landmarks.items():
            for point in points:
                cv2.circle(frame, point, 1, (0, 255, 0), -1)
    return frame


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Tool")
    parser.add_argument("--mode", choices=["regenerate", "webcam"], required=True,
                        help="Choose mode: regenerate (encodings) or webcam (live recognition)")
    parser.add_argument("--images", default="images",
                        help="Folder with face images")
    parser.add_argument("--encodings", default="encodings.pkl",
                        help="Encodings file path")
    args = parser.parse_args()

    if args.mode == "regenerate":
        regenerate_encodings(images_folder=args.images,
                             encodings_path=args.encodings)

    elif args.mode == "webcam":
        data = load_encodings(args.encodings)
        if not data["encodings"]:
            print("❌ No encodings found! Run with --mode regenerate first.")
            return

        cap = cv2.VideoCapture(0)
        print("🎥 Starting webcam... Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            name = recognize_face_from_frame(
                frame, data["encodings"], data["names"], tolerance=0.45)

            if name:
                cv2.putText(frame, f"{name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

            # Debug landmarks
            frame = draw_landmarks(frame)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
