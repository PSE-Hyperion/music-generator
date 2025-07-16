import logging

import numpy as np
from tensorflow.keras.utils import Sequence  # type: ignore - IGNORE ERROR, NOT ACTUAL ERROR

from music_generation_lstm.config import FEATURE_NAMES

logger = logging.getLogger(__name__)


class LazySequenceGenerator(Sequence):
    """
    Receives a list of .npz file paths,
    prepares internal indexing, optionally shuffles the order of samples, also prepares them for embedding

    Assumes, that all file paths are correct
    """

    def __init__(self, file_paths, batch_size=32, shuffle=True):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._build_sample_index()
        self.on_epoch_end()

        # Call super().__init__ to avoid a warning
        super().__init__()

    def _build_sample_index(self):
        """
        Stores file path and sample count for each file, builds a list of sample indices
        """

        self.data_info = []
        self.sample_map = []

        for file_idx, file_path in enumerate(self.file_paths):
            with np.load(file_path) as data:
                n_samples = len(data["y"])
                self.data_info.append((file_path, n_samples))
                for sample_idx in range(n_samples):
                    self.sample_map.append((file_idx, sample_idx))

        self.n_samples = len(self.sample_map)

    def __len__(self):
        """
        Returns how many batches exist per epoch
        """

        # Check if floor division ignores remaining batches
        return self.n_samples // self.batch_size

    def __getitem__(self, index):
        """
        Returns a batch of samples, given the index
        """

        batch_indices = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        x_batch, y_batch = [], []

        for sample_global_idx in batch_indices:
            file_idx, sample_idx = self.sample_map[sample_global_idx]
            file_path, _ = self.data_info[file_idx]

            with np.load(file_path) as data:
                x_batch.append(data["x"][sample_idx])
                y_batch.append(data["y"][sample_idx])

        x_array = np.array(x_batch)
        y_array = np.array(y_batch)

        # This splitting was previously done in train.py, but now needs to be done inside of the sequence generator

        # Split inputs into feature-wise dictionaries for multi-input model

        x_dict = {FEATURE_NAMES[i]: x_array[:, :, i] for i in range(len(FEATURE_NAMES))}
        """
        Creates a map similar to this:
        {
            'bar':      x_array[:, :, 0],
            'position': x_array[:, :, 1],
            'pitch':    x_array[:, :, 2],
            'duration': x_array[:, :, 3],
            'velocity': x_array[:, :, 4],
            'tempo':    x_array[:, :, 5]
        }
        Such that dict["bar"] only contains a 2d array with batch size * sequence length. Each entry is the bar at that
        sequence step in a batch.
        """

        # Split outputs into feature-wise arrays for multi-output model
        # Return as tuple of numpy arrays, instead of lists (I believe lists can't be input into a model)
        y_outputs = tuple(y_array[:, i] for i in range(6))

        return x_dict, y_outputs

    def on_epoch_end(self):
        """
        Shuffles the sample indices randomly at the end of each epoch
        """

        self.indexes = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)
