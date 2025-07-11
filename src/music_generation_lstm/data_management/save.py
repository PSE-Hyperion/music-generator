from collections.abc import Iterable
from pathlib import Path

import numpy as np

from music_generation_lstm import config
from music_generation_lstm.data_management import utils
from music_generation_lstm.processing.tokenization.tokens import BaseToken
from music_generation_lstm.step import pipeline_step


@pipeline_step(kind='saver')
def save_arrays_flat(input_data: Iterable[Iterable[BaseToken]], target: str) -> None:
    concat_arrays, starts, ends = utils.concat_arrays(map(utils.items_to_array_rowwise, input_data))
    np.savez_compressed(
        Path(config.PROCESSED_DIR).joinpath(target).with_suffix('.npz'),
        data=concat_arrays,
        starts=starts,
        ends=ends
    )
