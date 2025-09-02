import cv2
import numpy as np
import face_recognition

def recognize_face_from_frame(frame, known_encodings, known_names, tolerance=0.5):
    """
    Try to recognize a face in the given frame.

    - Downscales frame for speed.
    - Converts to RGB (since OpenCV uses BGR).
    - Compares against known encodings.

    Returns:
        name (str) if matched, else None
    """
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    if not face_locations:
        return None
    encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=tolerance)
        distances = face_recognition.face_distance(known_encodings, encoding)
        if True in matches:
            best_match_index = np.argmin(distances)
            return known_names[best_match_index]
    return None
