import logging
from multiprocessing import Pool, cpu_count
import os

from ..midi import parser
from . import process, processed_io
from .tokenization import token_map_io
from .tokenization.tokenizer import Sixtuple, SixtupleTokenMaps, Tokenizer

logger = logging.getLogger(__name__)


def _parallel_tokenize_worker(midi_path: str, processed_dataset_id: str) -> tuple[str, list[list[Sixtuple]]]:
    """
    A worker (per cpu core) tokenizing
        1. parse
        2. tokenize
    a single midi path.
    """

    tokenizer = Tokenizer(processed_dataset_id)

    # Parse
    parsed_midi = parser.parse_midi(midi_path)

    # Tokenize according to mode and type of parser
    song_versions = tokenizer.tokenize(parsed_midi)

    return (midi_path, song_versions)


def _parallel_process_worker(
    song_versions: list[list[Sixtuple]],
    sixtuple_token_maps: SixtupleTokenMaps,
    midi_path: str,
    processed_dataset_id: str,
) -> None:
    """
    A worker (per cpu core) processing
        1. numerize
        2. reshape
        3. save as .npz
    a single sixtuple list using the given (complete) sixtuple token maps
    """

    base_name = os.path.splitext(os.path.basename(midi_path))[0]

    for i, version_sixtuples in enumerate(song_versions):
        unique_name = f"{base_name}_original" if i == 0 else f"{base_name}_transpose{i}"

        # Numerize
        numeric_sixtuples = process.numerize(version_sixtuples, sixtuple_token_maps)

        # Save as continuous sequence instead of sequenizing
        continuous_sequence = process.create_continuous_sequence(numeric_sixtuples)

        # Save .npz with continuous sequence
        processed_io.save_continuous_data(processed_dataset_id, unique_name, continuous_sequence)


def parallel_process(dataset_id: str, processed_dataset_id: str) -> None:
    """
    Finds all midi paths in dataset, then starts worker pool and distributes each tokenize/process task of a single song
    to a different cpu core.
    """

    midi_paths = parser.get_midi_paths_from_dataset(dataset_id)

    with Pool(cpu_count()) as pool:
        results: list[tuple[str, list[list[Sixtuple]]]] = pool.starmap(
            _parallel_tokenize_worker, [(midi_path, processed_dataset_id) for midi_path in midi_paths]
        )

    sixtuple_token_maps = SixtupleTokenMaps()
    sixtuple_token_maps.create_from_ranges()

    with Pool(cpu_count()) as pool:
        pool.starmap(
            _parallel_process_worker,
            [
                (song_versions, sixtuple_token_maps, midi_path, processed_dataset_id)
                for midi_path, song_versions in results
            ],
        )

    token_map_io.save_token_maps(processed_dataset_id, sixtuple_token_maps)


def get_num_cores() -> int:
    return cpu_count()
