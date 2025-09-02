# Duplicate handling is already integrated in task5.py inside build_encodings()
# Example snippet:

def handle_duplicate(names, new_name):
    if new_name in names:
        print(f"⚠️ Duplicate user {new_name}, skipping.")
        return False
    return True
