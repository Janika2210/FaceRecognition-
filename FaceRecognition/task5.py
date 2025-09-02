import os, pickle, argparse, face_recognition

def build_encodings(image_dir, output="encodings.pkl"):
    encodings, names = [], []
    for file in os.listdir(image_dir):
        if not file.lower().endswith((".jpg", ".png")):
            continue
        path = os.path.join(image_dir, file)
        image = face_recognition.load_image_file(path)
        faces = face_recognition.face_encodings(image)
        if len(faces) != 1:
            print(f"⚠️ Skipping {file} (found {len(faces)} faces)")
            continue
        name = os.path.splitext(file)[0]
        if name in names:
            print(f"⚠️ Duplicate user {name}, skipping.")
            continue
        encodings.append(faces[0])
        names.append(name)
        print(f"[+] Added {name}")
    with open(output, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print(f"✅ Encodings saved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate face encodings")
    parser.add_argument("--images", required=True, help="Directory with user images")
    parser.add_argument("--output", default="encodings.pkl", help="Output pickle file")
    args = parser.parse_args()
    build_encodings(args.images, args.output)
