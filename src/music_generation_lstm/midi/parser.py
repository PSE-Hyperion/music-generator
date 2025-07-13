import glob
import os
import logging

from music21 import converter, stream

from music_generation_lstm.config import ALLOWED_MUSIC_FILE_EXTENSIONS, DATASETS_MIDI_DIR

logger = logging.getLogger(__name__)


def get_midi_paths_from_dataset(dataset_id: str) -> list[str]:
    #   Get all midi file paths in dataset
    #   This is used to avoid loading all files at same
    #   The returned files then can be used to process all songs of the dataset seperatly

    logger.info("Started parsing %s...", dataset_id)

    path = os.path.join(DATASETS_MIDI_DIR, dataset_id)

    midi_paths = []

    if os.path.isdir(path):
        for extension in ALLOWED_MUSIC_FILE_EXTENSIONS:
            midi_paths.extend(glob.glob(os.path.join(path, f"*{extension}")))
        total = len(midi_paths)
        logger.info("Folder found with %s accepted midi files.", total)
    elif os.path.isfile(path):
        if path.lower().endswith(tuple(ALLOWED_MUSIC_FILE_EXTENSIONS)):
            midi_paths.append(path)
            logger.info("File found")
        else:
            raise Exception("File found, but doesn't have allowed extension.")
    else:
        raise Exception("Invalid path.")

    return midi_paths


def parse_midi(midi_path: str) -> stream.Score:
    #   Parses music file to score using music21 converter
    #   Returns it, if the parsed result is a Score instance (not Opus or Part)
    #   Otherwise throws exceptions

    if os.path.isfile(midi_path):
        try:
            parsed = converter.parse(midi_path)
            if isinstance(parsed, stream.Score):
                return parsed
            raise Exception(
                "Parsed music file isn't Score."
            )  # Instead of exception, maybe ignore file and print warning
        except Exception as e:
            # Instead of exception, maybe ignore file and print warning
            raise Exception(f"Parsing of {midi_path} failed: {e}") from e
    else:
        raise Exception(f"Invalid path {midi_path}")
