from dataclasses import dataclass
import logging
from multiprocessing import Pool, cpu_count
import os

from ..midi import parser
from . import process, processed_io
from .tokenization import token_map_io
from .tokenization.tokenizer import Sixtuple, SixtupleTokenMaps, Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class SixtupleSets:
    bar_set: set[str]
    position_set: set[str]
    pitch_set: set[str]
    duration_set: set[str]
    velocity_set: set[str]
    tempo_set: set[str]


def _parallel_tokenize_worker(
    midi_path: str, processed_dataset_id: str
) -> tuple[str, list[list[Sixtuple]], SixtupleSets]:
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

    # Create sets of unique tokens (per feature)
    bar_set = set()
    position_set = set()
    pitch_set = set()
    duration_set = set()
    velocity_set = set()
    tempo_set = set()

    for version_sixtuples in song_versions:  # Iterate through each version
        for s in version_sixtuples:  # Iterate through sixtuples in each version
            bar_set.add(s.bar)
            position_set.add(s.position)
            pitch_set.add(s.pitch)
            duration_set.add(s.duration)
            velocity_set.add(s.velocity)
            tempo_set.add(s.tempo)

    return (
        midi_path,
        song_versions,
        SixtupleSets(bar_set, position_set, pitch_set, duration_set, velocity_set, tempo_set),
    )


def _parallel_process_worker(
    song_versions: list[list[Sixtuple]],
    sixtuple_token_maps: SixtupleTokenMaps,
    midi_path: str,
    processed_dataset_id: str,
):
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


def parallel_process(dataset_id: str, processed_dataset_id: str):
    """
    Finds all midi paths in dataset

    Start worker pool for tokenization (heavy task)

    Each worker does tokenization of a single score at the given midi path

    Collect worker sixtuples and create a complete sixtuple token maps instance

    Start worker pool for processing

    Each worker does processing and saving the processed result of a single sixtuple list
    """

    midi_paths = parser.get_midi_paths_from_dataset(dataset_id)

    with Pool(cpu_count()) as pool:
        results: list[tuple[str, list[list[Sixtuple]], SixtupleSets]] = pool.starmap(
            _parallel_tokenize_worker, [(midi_path, processed_dataset_id) for midi_path in midi_paths]
        )

    # Create sets of unique tokens (per feature, for all songs in dataset)
    bar_set = set()
    position_set = set()
    pitch_set = set()
    duration_set = set()
    velocity_set = set()
    tempo_set = set()

    for _, _, token_sets in results:
        bar_set |= token_sets.bar_set
        position_set |= token_sets.position_set
        pitch_set |= token_sets.pitch_set
        duration_set |= token_sets.duration_set
        velocity_set |= token_sets.velocity_set
        tempo_set |= token_sets.tempo_set

    sixtuple_token_maps = SixtupleTokenMaps()
    sixtuple_token_maps.create_from_sets(bar_set, position_set, pitch_set, duration_set, velocity_set, tempo_set)

    with Pool(cpu_count()) as pool:
        pool.starmap(
            _parallel_process_worker,
            [
                (song_versions, sixtuple_token_maps, midi_path, processed_dataset_id)
                for midi_path, song_versions, _ in results
            ],
        )

    #   Calls parallel_tokenize_worker
    #   Joins results to a single complete sixtuple token map instance
    #   Calls parallel_process_worker
    #   Saves sixtuple token maps

    token_map_io.save_token_maps(processed_dataset_id, sixtuple_token_maps)


def get_num_cores() -> int:
    return cpu_count()
