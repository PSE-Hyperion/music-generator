from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial, reduce
from typing import ClassVar

from music_generation_lstm.data_management.load import load_file_paths
from music_generation_lstm.midi.parser import m21_parse_midi_batch
from music_generation_lstm.processing.tokenization.tokenizer import tokenize_batch
from music_generation_lstm.step import Context


class BasePipeline(ABC):
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target
        self.context = Context()
        self.step_chain = _compose(
            _config_steps(self.STEPS)
        )

    @property
    @abstractmethod
    def STEPS(self):  # noqa: N802
        pass

    def run(self):
        return self.step_chain(self.source)


class Processing(BasePipeline):
    STEPS: ClassVar = [
        load_file_paths,
        m21_parse_midi_batch,
        tokenize_batch
    ]

def _config_steps(functions: list[Callable]) -> list[Callable]:
    configured_functions = []
    for idx, fn in enumerate(functions):
        configured_functions.append(
            partial(fn, step_count = idx + 1, max_step_count = len(functions))
        )
    return configured_functions

def _compose(functions: list[Callable]) -> Callable:
    """ Composes functions into a single function """
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)
