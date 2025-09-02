import hashlib

def hash_encoding(encoding):
    """
    Convert a 128-dim face encoding into a SHA256 hash string.
    (For future secure storage)
    """
    return hashlib.sha256(encoding.tobytes()).hexdigest()
