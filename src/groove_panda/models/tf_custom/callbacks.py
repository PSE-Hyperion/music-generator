import logging
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks  # type: ignore

logger = logging.getLogger(__name__)


class TerminalPrettyCallback(callbacks.Callback):
    """
    Custom callback class to use as callback in model.fit of a keras model.

    Replaces the standard training callbacks, that were hard to read (multiple features, both accuracy and loss)
    with more readable custom version.
    """

    COLUMN_WIDTH = 20
    EPOCH_COLUMN_WIDTH = 7
    TIME_COLUMN_WIDTH = 10
    LOSS_COLUMN_WIDTH = 10
    FLOAT_PRECISION = "3"

    def __init__(self):
        super().__init__()
        self._header_printed = False

    def on_train_begin(self, logs=None) -> None:  # noqa: ARG002
        """
        Called once before the first epoch to log training start.
        """
        logger.info("Started training...")

    def on_epoch_begin(self, epoch, logs=None):  # noqa: ARG002
        """
        Called on epoch begin to save epoch start time
        """
        self._start_time = time.time()

    def on_epoch_end(self, epoch, logs: dict[str, float] | None = None) -> None:
        """
        Automatically gets called in model.fit of keras model during training, at the end of an epoch.

        Prints (logs) the training result of this epoch in a more readable way than standard keras callback.

        Preview:
            ```
            Epoch | Loss | Pitch | Duration | Position | Bar | Bar | Tempo | Velocity
            0     | 16   | 0.1   | 0.1      | 0.1      | 0.1 | 0.1 | 0.1   | 0.1
            1     | 12   | 0.2   | 0.1      | 0.2      | 0.2 | 0.2 | 0.2   | 0.2
            2     | 10   | 0.3   | 0.3        | 0.3    | 0.3 | 0.3 | 0.3   | 0.3
            3     | 8    | 0.4   | 0.4      | 0.4      | 0.4 | 0.4 | 0.4   | 0.4
            ```
        """

        logs = logs or {}

        outputs = sorted({key.split("_")[0] for key in logs if key != "loss" and "_" in key})

        header = [
            f"{'Epoch':<{TerminalPrettyCallback.EPOCH_COLUMN_WIDTH}}",
            f"{'Time':<{TerminalPrettyCallback.TIME_COLUMN_WIDTH}}",
            f"{'Loss':<{TerminalPrettyCallback.LOSS_COLUMN_WIDTH}}",
        ]

        for output in outputs:
            label = output.split("_")[0].capitalize()
            header.append(f"{label:<{TerminalPrettyCallback.COLUMN_WIDTH}}")

        end_time = time.time()
        duration = end_time - self._start_time
        mins, secs = divmod(duration, 60)

        values = [
            f"{epoch + 1:<{TerminalPrettyCallback.EPOCH_COLUMN_WIDTH}}",
            f"{f'{int(mins)}m {int(secs)}s':<{TerminalPrettyCallback.TIME_COLUMN_WIDTH}}",
            f"{logs.get('loss', 0):<{TerminalPrettyCallback.LOSS_COLUMN_WIDTH}.{TerminalPrettyCallback.FLOAT_PRECISION}f}",
        ]
        for output in outputs:
            loss_key = f"{output}_output_loss"
            acc_key = f"{output}_output_accuracy"
            if output == "loss":
                loss_key = "loss"
                acc_key = ""
            loss = logs.get(loss_key, 0.0)
            acc = logs.get(acc_key, 0.0)
            values.append(
                f"{loss:.{TerminalPrettyCallback.FLOAT_PRECISION}f}, {acc:.{TerminalPrettyCallback.FLOAT_PRECISION}f}".ljust(
                    TerminalPrettyCallback.COLUMN_WIDTH
                )
            )

        # print header once on first epoch for visual clarity (makes the output appear like a table)
        if not self._header_printed:
            print(" | ".join(header))
            self._header_printed = True

        print(" | ".join(values))


class EmbeddingPropertiesCallback(tf.keras.callbacks.Callback):
    """
    This is a custom callback for e.g. Tensorboard.
    It gives helpful information about a specified embedding layer for experiments and analysis of an architecture.
    Is not intended to be a default callback for all trainings.
    """
    def __init__(self, log_dir, layer_name='embedding', threshold=0.99):
        super().__init__()
        self.log_dir = log_dir
        self.layer_name = layer_name
        self.threshold = threshold
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, 'embedding_svd'))

    def log_singular_values(self, weights, step_label, step):
        u, s, vh = np.linalg.svd(weights, full_matrices=False)

        squared_s = s ** 2
        total_energy = np.sum(squared_s)
        energy_ratio = np.cumsum(squared_s) / total_energy
        effective_rank = np.searchsorted(energy_ratio, self.threshold) + 1
        p = squared_s / total_energy
        eps = 1e-12
        entropy = -np.sum(p * np.log(p + eps))
        entropy_rank = np.exp(entropy)

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
