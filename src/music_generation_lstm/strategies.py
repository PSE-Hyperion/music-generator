from functools import partial
from typing import Literal

from music_generation_lstm.config import MIDI_FILE_PATTERN, PROCESSED_DATASET_DIR, RAW_DATASET_DIR
from music_generation_lstm.data_management.load import load_file_paths
from music_generation_lstm.midi.parser import parse_midi
from music_generation_lstm.processing.process import numerize, sequenize
from music_generation_lstm.processing.processed_io import save_processed_data
from music_generation_lstm.processing.tokenization import tokenizer as sixtuple_tokenizer


def loader(strat: Literal['lazy-midi-paths'], source):
    strategies = {
        'lazy-midi-paths': partial(
                load_file_paths,
                working_dir = RAW_DATASET_DIR,
                dir_name = source,
                file_type_pattern = MIDI_FILE_PATTERN
        )
    }
    return strategies.get(strat)

def parser(strat: Literal['m21']):
    strategies = {
        'm21': parse_midi
    }
    return strategies.get(strat)

def tokenizer(strat: Literal['sixtuple']):
    strategies = {
        'sixtuple': sixtuple_tokenizer
    }
    return strategies.get(strat)

def encoder(strat: Literal['index-sixtupe']):
    strategies = {
        'index-sixtuple': numerize
    }
    return strategies.get(strat)

def trainig_preparation(strat: Literal['none', 'sequences']):
    strategies = {
        'none': (lambda x: x),
        'sequences': sequenize
    }
    return strategies.get(strat)

def saver(strat: Literal['training-sequences'], target: str):
    strategies = {
        'training-sequences': partial(
                save_processed_data,
                processed_dataset_id = target,
                music_path = PROCESSED_DATASET_DIR
        )
    }
    return strategies.get(strat)
