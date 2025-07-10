from collections.abc import Iterable
from pathlib import Path

from music_generation_lstm import config
from music_generation_lstm.step import pipeline_step


@pipeline_step(kind='loader')
def load_file_paths(dir_name: str) -> Iterable[Path]:
    """
    Lads recursively all files with the specified pattern from the specified dir.
    """
    path = Path(config.DATASETS_MIDI_DIR).joinpath(dir_name)
    return filter(
        lambda x: x.is_file and config.MIDI_FILE_PATTERN.fullmatch(x.name),
        path.rglob('*')
    )




