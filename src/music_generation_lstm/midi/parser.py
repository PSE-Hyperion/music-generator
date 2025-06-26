# load midi file(s) from given path and return them converted to music21 score
import os
import glob
from music21 import converter, stream
from config import RAW_MIDI_DIR, ALLOWED_MUSIC_FILE_EXTENSIONS



def get_midi_paths_from_dataset(dataset_id : str) -> list[str]:
    #   get all midi file paths in dataset
    #   this is used to avoid loading all files at same
    #   the returned files then can be used to process all songs of the dataset seperatly

    print(f"Started parsing at {dataset_id}")

    path = os.path.join(RAW_MIDI_DIR, dataset_id)

    midi_paths = []

    if os.path.isdir(path):
        for extension in ALLOWED_MUSIC_FILE_EXTENSIONS:
            midi_paths.extend(glob.glob(os.path.join(path, f"*{extension}")))
        total = len(midi_paths)
        print(f"Folder found with {total} accepted midi files.")
    elif os.path.isfile(path):
        if path.lower().endswith(tuple(ALLOWED_MUSIC_FILE_EXTENSIONS)):
            midi_paths.append(path)
            print("File found")
        else:
            raise Exception("File found, but doesn't have allowed extension.")
    else:
        raise Exception("Invalid path.")

    print(midi_paths)
    return midi_paths

def parse_midi(music_path : str) -> stream.Score:
    #   parses music file to score
    #
    #

    if os.path.isfile(music_path):
        try:
            parsed = converter.parse(music_path)
            if isinstance(parsed, stream.Score):
                return parsed
            else:
                raise Exception("Parsed music file isn't Score.")   # Instead of exception, maybe ignore file and print warning
        except Exception as e:
            raise Exception(f"Parsing of {music_path} failed: {e}") # Instead of exception, maybe ignore file and print warning
    else:
        raise Exception(f"Invalid path {music_path}")



