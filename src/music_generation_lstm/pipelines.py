from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial, reduce
import inspect
from logging import getLogger
from typing import ClassVar

from music_generation_lstm.data_management.load import load_midi_file_paths
from music_generation_lstm.data_management.save import save_arrays_flat
from music_generation_lstm.midi.parser import m21_parse_midi_batch
from music_generation_lstm.processing.tokenization.tokenizer import tokenize_batch
from music_generation_lstm.step import Context

logger = getLogger(__name__)

class BasePipeline(ABC):
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target
        self.context = Context()
        self.step_chain = _compose(
            _config_steps(
                self.STEPS,
                self.source,
                self.target
            )
        )

    @property
    @abstractmethod
    def STEPS(self):  # noqa: N802
        pass

    def run(self):
        try:
            return self.step_chain(None)
        except Exception as e:
            logger.error("Pipeline execution failed: %s", e)


class Processing(BasePipeline):
    STEPS: ClassVar = [
        load_midi_file_paths,
        m21_parse_midi_batch,
        tokenize_batch,
        save_arrays_flat
    ]

def _config_steps(functions: list[Callable], source: str, target: str) -> list[Callable]:
    configured_functions = []
    for idx, fn in enumerate(functions):
        sig = inspect.signature(fn)
        set_fn: Callable = fn
        if 'source' in sig.parameters:
            set_fn = partial(set_fn, source=source)
        if 'target' in sig.parameters:
            set_fn = partial(set_fn, target=target)
        if 'context' in sig.parameters:
            set_fn = partial(set_fn, context=Context())
        set_fn = partial(set_fn, step_count = idx + 1, max_step_count = len(functions))
        configured_functions.append(set_fn)
    return configured_functions

def _compose(functions: list[Callable]) -> Callable:
    """ Composes functions into a single function """
    return reduce(lambda f,g: lambda x: g(f(x)), functions)
