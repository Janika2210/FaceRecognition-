import cv2
import pickle
import face_recognition
import webbrowser
import os

ENCODING_FILE = os.path.join("encodings", "encodings.pkl")

def load_encodings():
    with open(ENCODING_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def recognize_face_from_frame(frame, known_encodings, known_names, tolerance=0.5):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance)
        if True in matches:
            idx = matches.index(True)
            return known_names[idx]
    return None

def main():
    if not os.path.exists(ENCODING_FILE):
        print("[ERROR] No encodings found. Run face_manager.py first.")
        return

    known_encodings, known_names = load_encodings()
    print("[INFO] Loaded encodings. Starting webcam...")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        name = recognize_face_from_frame(frame, known_encodings, known_names)

        cv2.putText(frame, f"User: {name if name else 'Unknown'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if name else (0, 0, 255), 2)

        cv2.imshow("Face Verification", frame)

        if name:
            print(f"[INFO] Verified: {name}. Opening Gmail...")
            webbrowser.open("https://mail.google.com")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
