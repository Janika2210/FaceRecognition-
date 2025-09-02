def build_encodings(image_dir, output="encodings.pkl"):
    """
    Scan a directory of face images and generate encodings.

    Parameters:
        image_dir (str): Folder with .jpg/.png files. Each filename = user name.
        output (str): Where to save the pickle file.

    Notes:
        - Skips images with zero or multiple faces.
        - Ignores duplicate names.
        - Output pickle contains a dict { "encodings": list, "names": list }.
    """
    pass  # actual code is in task5.py
