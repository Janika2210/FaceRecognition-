import cv2
import numpy as np
import face_recognition
from task1 import recognize_face_from_frame

def recognize_with_confidence(frame, known_encodings, known_names, tolerance=0.5, distance_threshold=0.55):
    """
    Adds an extra confidence check using face distance.
    """
    name = recognize_face_from_frame(frame, known_encodings, known_names, tolerance)
    if not name:
        return None
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    encoding = face_recognition.face_encodings(rgb_small)[0]
    distances = face_recognition.face_distance(known_encodings, encoding)
    min_dist = np.min(distances)
    return name if min_dist < distance_threshold else None
