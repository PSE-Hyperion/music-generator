import logging

import numpy as np
from tensorflow.keras.utils import Sequence  # type: ignore - IGNORE ERROR, NOT ACTUAL ERROR

from groove_panda.config import FEATURE_NAMES

logger = logging.getLogger(__name__)


class FlexibleSequenceGenerator(Sequence):
    """
    Generator that extracts sequences of configurable length from continuous song data

    Receives a list of .npz file paths with continuous sequences,
    prepares internal indexing, optionally shuffles the order of samples, also prepares them for embedding

    Assumes, that all file paths are correct and contain continuous_sequence data
    """

    def __init__(
        self, file_paths: list[str], sequence_length: int, stride: int = 1, batch_size: int = 32, shuffle: bool = True
    ):
        self.file_paths = file_paths
        self.sequence_length = sequence_length
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._load_continuous_data()
        self._build_sample_index()
        self.on_epoch_end()

        # Call super().__init__ to avoid a warning
        super().__init__()

    def _build_sample_index(self):
        """Build deterministic index of all possible subsequences"""
        self.sample_map = []

        for song_idx, song_data in enumerate(self.song_data):
            continuous_seq = song_data[0]  # Only get the sequence, ignore possible_samples
            max_start_idx = len(continuous_seq) - self.sequence_length - 1
            max_start_steps = max_start_idx // self.stride

            for start_step in range(max_start_steps + 1):
                start_idx = start_step * self.stride
                self.sample_map.append((song_idx, start_idx))

    def _load_continuous_data(self):
        """
        Load all continuous sequences and calculate total possible samples
        """
        self.song_data = []
        self.total_samples = 0

        for file_path in self.file_paths:
            with np.load(file_path) as data:
                if "continuous_sequence" not in data:
                    logger.warning("File %s does not contain continuous_sequence data", file_path)
                    continue

                continuous_seq = data["continuous_sequence"]
                if len(continuous_seq) >= self.sequence_length + 1:
                    # Calculate possible samples with stride
                    max_start_idx = len(continuous_seq) - self.sequence_length - 1  # -1 for the target y
                    possible_samples = (max_start_idx // self.stride) + 1
                    self.song_data.append((continuous_seq, possible_samples))
                    self.total_samples += possible_samples
                else:
                    logger.warning(
                        "Sequence in %s too short: %d < %d", file_path, len(continuous_seq), self.sequence_length + 1
                    )

    def __len__(self):
        """
        Returns how many batches exist per epoch
        """
        return self.total_samples // self.batch_size

    def __getitem__(self, index):
        """Extract subsequences deterministically"""
        batch_x, batch_y = [], []

        batch_indices = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        for sample_idx in batch_indices:
            song_idx, start_idx = self.sample_map[sample_idx]
            continuous_seq, _ = self.song_data[song_idx]

            x, y = self._extract_subsequence(continuous_seq, start_idx)
            batch_x.append(x)
            batch_y.append(y)

        return self._format_batch(batch_x, batch_y)

    def _extract_subsequence(self, sequence: np.ndarray, start_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract subsequence starting at given index
        """
        x = sequence[start_idx : start_idx + self.sequence_length]
        y = sequence[start_idx + self.sequence_length]
        return x, y

    def _format_batch(self, batch_x: list, batch_y: list) -> tuple[dict, tuple]:
        """
        Format batch for model input
        """
        x_array = np.array(batch_x)
        y_array = np.array(batch_y)

        # Split inputs into feature-wise dictionaries for multi-input model
        x_dict = {f"input_{FEATURE_NAMES[i]}": x_array[:, :, i] for i in range(len(FEATURE_NAMES))}

        # Split outputs into feature-wise arrays for multi-output model
        y_outputs = tuple(y_array[:, i] for i in range(6))

        return x_dict, y_outputs

    def on_epoch_end(self):
        """Shuffle sample indices at epoch end"""
        self.indexes = np.arange(len(self.sample_map))
        if self.shuffle:
            np.random.shuffle(self.indexes)
