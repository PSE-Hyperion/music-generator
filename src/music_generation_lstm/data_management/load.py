from collections.abc import Iterable
from pathlib import Path

from music_generation_lstm import config
from music_generation_lstm.step import pipeline_step

"""
Class for loading data.
Used for abstraction from the file system.
"""

@pipeline_step(kind='loader')
def load_midi_file_paths(dir_name: str) -> Iterable[Path]:
    """
    Lads recursively all files with the specified pattern from the specified dir.
    Functional design for lazy loading of paths (not necessary to store them all at once).
    """
    working_dir = config.DATASETS_MIDI_DIR
    file_pattern = config.MIDI_FILE_PATTERN
    parent_path_of_files = Path(working_dir).joinpath(dir_name)

    # Works from last line to first line
    # 1. Loads all files and dirs in the dir and all sub dirs recursively
    # 2. Filters the ones that are files and match the pattern of midi files '*.mid' or '*.midi'
    return filter(
        lambda x: x.is_file and file_pattern.fullmatch(x.name),
        parent_path_of_files.rglob('*')
    )


