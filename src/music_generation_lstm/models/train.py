from pyexpat import features

import tensorflow as tf
import numpy as np
import logging

from tensorflow.keras.callbacks import History  # type: ignore

from music_generation_lstm.config import TRAINING_BATCH_SIZE, TRAINING_EPOCHS
from music_generation_lstm.models import plot
from music_generation_lstm.models.lazy_sequence_generator import LazySequenceGenerator
from music_generation_lstm.models.models import BaseModel
from music_generation_lstm.models.training_callback import TrainingCallback

logger = logging.getLogger(__name__)


def train_model(model: BaseModel, file_paths: list):
    """
    Train model using LazySequenceGenerator for memory-efficient training on large datasets.

    File paths contain the paths to .npz files. Needed to create a lazy sequence generator, that will lazily load
    samples in batches (and random samples and shuffles)
    """

    logger.info("Start training with lazy loading...")

    try:
        train_generator = LazySequenceGenerator(file_paths=file_paths, batch_size=TRAINING_BATCH_SIZE, shuffle=True)

        steps_per_epoch = len(train_generator)

        logger.info("Training with %s files, containing %s total samples", len(file_paths), train_generator.n_samples)
        logger.info("Steps per epoch: %s, Batch size: %s", steps_per_epoch, TRAINING_BATCH_SIZE)

        # fit() will automatically call on_epoch_end of lazy sequence generator, to get new samples
        history = model.model.fit(
            train_generator,
            epochs=TRAINING_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            verbose=0,  # type: ignore
            callbacks=[TrainingCallback()],
            # Note: validation_split doesn't work with generators,
            # you'd need a separate validation generator (or other solution)
        )

    except Exception as e:
        raise Exception(f"Training failed: {e}") from e

    logger.info("Finished training %s", model.model_id)

    if isinstance(history, History):
        plot.plot_training(history, model.model_id)


def train_model_eager(model: BaseModel, file_paths: list):
    """
    Loads all necessary data upfront.
    This reduces the latency per sample. The model can take the samples directly from the RAM and doesn't have to load a file.
    It leads to better use of the computation units resources.
    It should be the most efficient way of training until the size of the dataset is at least 1 GB
    """
    try:
        full_list_X = []
        full_list_y = []
        # Collects all batches in one list
        for path in file_paths:
            data = np.load(path)
            full_list_X.append(data["X"])
            full_list_y.append(data["y"])
        # Conversion into numpy arrays
        full_array_X = np.concatenate(full_list_X)
        full_array_y = np.concatenate(full_list_y)
        # Assert that the formats of X and y arrays match
        assert full_array_X.shape[0] == full_array_y.shape[0]
        dataset_size = full_array_X.shape[0]

        # Conversion for the model
        X_dict = {
            feature: full_array_X[:, :, idx]
            for idx, feature in enumerate(["bar", "position", "pitch", "duration", "velocity", "tempo"])
        }
        y_output = tuple(
            full_array_y[:, idx]
            for idx, feature in enumerate(["bar", "position", "pitch", "duration", "velocity", "tempo"])
        )
        dataset = tf.data.Dataset.from_tensor_slices((X_dict, y_output))

        # Providing input for the model is now handled by Tensorflow since it's maximally optimized
        # Shuffle all samples
        dataset = dataset.shuffle(buffer_size=dataset_size)
        # Creating new batches of the data
        dataset = dataset.batch(TRAINING_BATCH_SIZE)
        # Automates how TF prefetches the batches for better resource use
        # Since we already tell TF to shuffle all samples and the samples are all stored in the dict in the RAM,
        # this could have no effect at all (maybe on GPU training)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        history = model.model.fit(dataset, epochs=TRAINING_EPOCHS, verbose=2)
    except Exception as e:
        raise Exception(f"Training failed: {e}").with_traceback(e)
