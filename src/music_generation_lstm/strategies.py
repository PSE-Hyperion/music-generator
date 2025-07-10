from functools import partial
from typing import Literal

from music_generation_lstm.config import DATASETS_MIDI_DIR, MIDI_FILE_PATTERN
from music_generation_lstm.data_management.load import load_file_paths
from music_generation_lstm.midi.parser import m21_parse_midi_batch
from music_generation_lstm.processing.encoder import tokens_to_arrays
from music_generation_lstm.processing.process import sequenize
from music_generation_lstm.processing.processed_io import save_processed_data
from music_generation_lstm.processing.tokenization import tokenizer as sixtuple_tokenizer


def loader(strat: Literal['lazy-midi-paths']):
    strategies = {
        'lazy-midi-paths': partial(
                load_file_paths,
                working_dir = DATASETS_MIDI_DIR,
                file_pattern = MIDI_FILE_PATTERN
        )
    }
    return strategies.get(strat)

def parser(strat: Literal['m21']):
    strategies = {
        'm21': m21_parse_midi_batch
    }
    return strategies.get(strat)

def tokenizer(strat: Literal['sixtuple']):
    strategies = {
        'sixtuple': sixtuple_tokenizer.Tokenizer("DUMMY")
    }
    return strategies.get(strat)

def encoder(strat: Literal['index-sixtupe']):
    strategies = {
        'index-sixtuple': tokens_to_arrays
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
                music_path = DATASETS_MIDI_DIR
        )
    }
    return strategies.get(strat)
