from collections.abc import Iterable
from pathlib import Path
from re import Pattern


def load_file_paths(dir_name, working_dir: str, file_pattern: Pattern[str]) -> Iterable[Path]:
    """
    Lads recursively all files with the specified pattern from the specified dir.
    """

    path = Path(working_dir).joinpath(dir_name)
    return filter(
        lambda x: x.is_file and file_pattern.fullmatch(x.name),
        path.rglob('*')
    )



