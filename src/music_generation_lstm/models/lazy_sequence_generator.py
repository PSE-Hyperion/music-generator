from keras.api.utils import Sequence
import numpy as np

class LazySequenceGenerator(Sequence):

    # Receives a list of .npz file paths, prepares internal indexing, optionally shuffles the order of samples, also prepares them for embedding
    # Assumes, that all file paths are correct
    def __init__(self, file_paths, batch_size=32, shuffle=True):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._build_sample_index()
        self.on_epoch_end()

        # Call super().__init__ to avoid the warning
        super().__init__()

    # Stores file path and sample count for each file, builds a list of sample indices
    def _build_sample_index(self):

        self.data_info = []
        self.sample_map = []

        for file_idx, file_path in enumerate(self.file_paths):
            with np.load(file_path) as data:
                n_samples = len(data['y'])
                self.data_info.append((file_path, n_samples))
                for sample_idx in range(n_samples):
                    self.sample_map.append((file_idx, sample_idx))

        self.n_samples = len(self.sample_map)

    # Returns how many batches exist per epoch
    def __len__(self):
        return self.n_samples // self.batch_size

    # Returns a batch of samples, given the index
    def __getitem__(self, index):
        batch_indices = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        x_batch, y_batch = [], []

        for sample_global_idx in batch_indices:
            file_idx, sample_idx = self.sample_map[sample_global_idx]
            file_path, _ = self.data_info[file_idx]

            with np.load(file_path) as data:
                x_batch.append(data['X'][sample_idx])
                y_batch.append(data['y'][sample_idx])

        x_array = np.array(x_batch)
        y_array = np.array(y_batch)

        # This splitting was previously done in train.py, but now needs to be done inside of the sequence generator

        # Split inputs into feature-wise dictionaries for multi-input model
        feature_names = ["bar", "position", "pitch", "duration", "velocity", "tempo"]
        x_dict = {feature_names[i]: x_array[:, :, i] for i in range(6)}

        # Split outputs into feature-wise arrays for multi-output model
        # Return as tuple of numpy arrays, not list
        y_outputs = tuple(y_array[:, i] for i in range(6))

        return x_dict, y_outputs

    # Shuffles the sample indices randomly at the end of each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)
