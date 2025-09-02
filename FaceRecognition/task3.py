import logging
logging.basicConfig(filename="recognition.log", level=logging.INFO)

def log_result(name, distances):
    if name:
        logging.info(f"[MATCH] {name} | Distances: {distances}")
    else:
        logging.info("[NO MATCH] No recognized face in frame.")
