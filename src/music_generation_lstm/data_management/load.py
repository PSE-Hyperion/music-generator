from collections.abc import Iterable
from pathlib import Path


def load_file_paths(working_dir: str, dir_name: str, file_type_pattern: str) -> Iterable[str]:
    """ Lads recursively all files with the specified pattern from the specified dir. """
    path = Path(working_dir).joinpath(dir_name)
    unfiltered_file_paths = path.rglob("*.midi?")

    return filter(
        lambda x: x.is_file(),
        unfiltered_file_paths
        )

