# Face Authentication for Gmail

## Overview
This project recognizes a user's face using the webcam and, if verified, automatically opens Gmail.

## Structure
```
face_auth_gmail/
│── data/                   # Store known user images (.jpg/.png)
│── encodings/              # Store encodings.pkl
│── src/
│   │── face_manager.py     # Add new faces & regenerate encodings
│   │── face_auth.py        # Verify face & open Gmail
│── requirements.txt
│── README.md
```

## Steps to Run
1. Add face images into `data/` (each image should have only one face).
2. Run `python src/face_manager.py` to generate encodings.
3. Run `python src/face_auth.py` to verify and open Gmail.
