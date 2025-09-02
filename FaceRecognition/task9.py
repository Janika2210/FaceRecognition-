import face_recognition

def validate_single_face(image_path):
    """
    Ensure exactly 1 face in an image.
    """
    img = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(img)
    if len(encodings) != 1:
        raise ValueError(f"{image_path} has {len(encodings)} faces (expected 1)")
    return encodings[0]
