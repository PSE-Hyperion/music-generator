from tensorflow.keras.utils import Sequence
import numpy as np
import os

class LazySequenceGenerator(Sequence):

    #Receives a list of .npz file paths, prepares internal indexing, optionally shuffles the order of samples
    def __init__(self, file_paths, batch_size=32, shuffle=True):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._build_sample_index()
        self.on_epoch_end()

    #Stores file path and sample count for each file, builds a list of sample indices
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

    #Returns how many batches exsit per epoch
    def __len__(self):
        return self.num_samples // self.batch_size

    #Returns a batch of samples, given the index
    def __getitem__(self, index):
        batch_indices = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        x_batch, y_batch = [], []

        for sample_global_idx in batch_indices:
            file_idx, sample_idx = self.sample_map[sample_global_idx]
            file_path, _ = self.data_info[file_idx]

            with np.load(file_path) as data:
                x_batch.append(data['X'][sample_idx])
                y_batch.append(data['y'][sample_idx])

        return np.array(x_batch), np.array(y_batch)

    #Shuffles the sample indices randomly at the end of each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)
