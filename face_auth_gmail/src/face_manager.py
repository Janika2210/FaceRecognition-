import os
import pickle
import face_recognition

DATA_DIR = os.path.join("data")
ENCODING_FILE = os.path.join("encodings", "encodings.pkl")

def generate_encodings():
    known_encodings = []
    known_names = []

    for file_name in os.listdir(DATA_DIR):
        if file_name.lower().endswith((".jpg", ".png")):
            path = os.path.join(DATA_DIR, file_name)
            name = os.path.splitext(file_name)[0]

            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 1:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"[INFO] Added encoding for {name}")
            else:
                print(f"[ERROR] {file_name}: Image must contain exactly 1 face.")

    with open(ENCODING_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    print("[INFO] Encodings saved to", ENCODING_FILE)


if __name__ == "__main__":
    os.makedirs("encodings", exist_ok=True)
    generate_encodings()
