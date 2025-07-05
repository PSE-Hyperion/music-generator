import glob
import os

from music21 import converter, stream

from ..config import ALLOWED_MUSIC_FILE_EXTENSIONS, DATASETS_MIDI_DIR


def get_midi_paths_from_dataset(dataset_id: str) -> list[str]:
    #   Get all midi file paths in dataset
    #   This is used to avoid loading all files at same
    #   The returned files then can be used to process all songs of the dataset seperatly

    print(f"Started parsing {dataset_id}...")

    path = os.path.join(DATASETS_MIDI_DIR, dataset_id)

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

    return midi_paths


def parse_midi(music_path: str) -> stream.Score:
    #   Parses music file to score using music21 converter
    #   Returns it, if the parsed result is a Score instance (not Opus or Part)
    #   Otherwise throws exceptions

    if os.path.isfile(music_path):
        try:
            parsed = converter.parse(music_path)
            if isinstance(parsed, stream.Score):
                return parsed
            else:
                raise Exception(
                    "Parsed music file isn't Score."
                )  # Instead of exception, maybe ignore file and print warning
        except Exception as e:
            raise Exception(
                f"Parsing of {music_path} failed: {e}"
            )  # Instead of exception, maybe ignore file and print warning
    else:
        raise Exception(f"Invalid path {music_path}")
