import cv2
import face_recognition

def draw_face_landmarks(frame):
    """
    Draw small green dots on facial landmarks (eyes, nose, lips, etc.)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    all_landmarks = face_recognition.face_landmarks(rgb)
    for face_landmarks in all_landmarks:
        for feature, points in face_landmarks.items():
            for (x, y) in points:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    return frame
