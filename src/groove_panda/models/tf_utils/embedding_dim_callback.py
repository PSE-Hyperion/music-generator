import tensorflow as tf
import numpy as np
import os


class EmbeddingSVDLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, layer_name='embedding', threshold=0.99):
        super().__init__()
        self.log_dir = log_dir
        self.layer_name = layer_name
        self.threshold = threshold
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, 'embedding_svd'))

    def log_singular_values(self, weights, step_label, step):
        u, s, vh = np.linalg.svd(weights, full_matrices=False)

        # Effektiver Rang berechnen (PCA-Kriterium)
        squared_s = s ** 2
        total_energy = np.sum(squared_s)
        energy_ratio = np.cumsum(squared_s) / total_energy
        effective_rank = np.searchsorted(energy_ratio, self.threshold) + 1
        p = squared_s / total_energy
        eps = 1e-12
        entropy = -np.sum(p * np.log(p + eps))
        entropy_rank = np.exp(entropy)

        # Statistiken
        spectral_norm = s[0]  # Größter Singulärwert
        nuclear_norm = np.sum(s)  # Summe aller Singulärwerte
        num_above_thresh = np.sum(s > 0.1)

        with self.writer.as_default():
            tf.summary.scalar(f"{step_label}/EffectiveRank", effective_rank, step=step)
            tf.summary.scalar(f"{step_label}/EntropyRank", entropy_rank, step=step)
            tf.summary.histogram(f"{step_label}/SingularValues", s, step=step)
            tf.summary.scalar(f"{step_label}/SpectralNorm", spectral_norm, step=step)
            tf.summary.scalar(f"{step_label}/NuclearNorm", nuclear_norm, step=step)
            tf.summary.scalar(f"{step_label}/NumSingularValues>0.1", num_above_thresh, step=step)
        self.writer.flush()

    def on_train_begin(self, logs=None):
        layer = self._get_embedding_layer()
        if layer is not None:
            weights = layer.get_weights()[0]
            self.log_singular_values(weights, "BeforeTraining", step=0)

    def on_epoch_end(self, epoch, logs=None):
        layer = self._get_embedding_layer()
        if layer is not None:
            weights = layer.get_weights()[0]
            self.log_singular_values(weights, "Epoch", step=epoch)

    def _get_embedding_layer(self):
        for l in self.model.layers:
            if self.layer_name in l.name:
                return l
        print(f"[SVD Callback] Layer '{self.layer_name}' not found.")
        return None
