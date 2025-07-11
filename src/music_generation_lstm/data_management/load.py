from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from music_generation_lstm import config
from music_generation_lstm.step import pipeline_step

"""
Class for loading data.
Used for abstraction from the file system.
"""

@pipeline_step(kind='loader')
def load_midi_file_paths(_, source: str) -> Iterable[Path]:
    """
    Lads recursively all files with the specified pattern from the specified dir.
    Functional design for lazy loading of paths (not necessary to store them all at once).
    """
    working_dir = config.DATASETS_MIDI_DIR
    file_pattern = config.MIDI_FILE_PATTERN
    parent_path_of_files = Path(working_dir).joinpath(source)

    # Works from last line to first line
    # 1. Loads all files and dirs in the dir and all sub dirs recursively
    # 2. Filters the ones that are files and match the pattern of midi files '*.mid' or '*.midi'
    return filter(
        lambda x: x.is_file and file_pattern.fullmatch(x.name),
        parent_path_of_files.rglob('*')
    )

def load_dataset(_, source: str) -> tuple[NDArray, NDArray, NDArray]:
    loaded = np.load(Path(config.PROCESSED_DIR).joinpath(source).with_suffix('.npz'), mmap_mode='r')
    data = loaded["data"]
    starts = loaded["starts"]
    ends = loaded["ends"]

    return data, starts, ends
