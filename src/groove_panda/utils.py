import json
import os


def overwrite_json(file_path: str, data):
    """
    Checks if an outdated version of the data to be saved exists and overwrites it if it's the case.
    If no data is present, the provided data is simply saved.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "w") as fp:
        json.dump(data, fp, indent=2)  # Indent = 2 for readability
