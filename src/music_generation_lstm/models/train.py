import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import History  # type: ignore

from music_generation_lstm.config import FEATURE_NAMES, TRAINING_BATCH_SIZE, TRAINING_EPOCHS
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
            verbose=2,  # type: ignore
            # callbacks=[TrainingCallback()],
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
    This reduces the latency per sample. The model can take the samples
    directly from the RAM and doesn't have to load a file.
    It leads to better use of the computation units resources.
    It should be the most efficient way of training until the size of the dataset is at least 1 GB
    """

    try:
        logger.info("Start gathering processed songs...")
        # Concatenates the sample seq["bar", "position", "pitch", "duration", "velocity", "tempo"])uences of all files
        # into an array
        full_array_x = np.concatenate([(np.load(path))["x"] for path in file_paths])
        # Concatenates the sample targets of all files into an array
        full_array_y = np.concatenate([(np.load(path))["y"] for path in file_paths])
        # Ensures that the lengths of samples of targets match
        assert full_array_x.shape[0] == full_array_y.shape[0]
        dataset_size = full_array_x.shape[0]

        # Structure of the arrays:
        # sample sequences x: [<sample_idx>, <token_in_sample_idx>, <feature_idx>]
        # sample targets: [<sample_idx>, <feature_idx>]

        logger.info("Start converting...")
        # Conversion for the model input layers
        # Iterating over the feature axis of the tensors
        x_dict = {
            feature: full_array_x[:, :, idx]  # take of each sample only the specified feature
            for idx, feature in enumerate(FEATURE_NAMES)
        }
        y_output = tuple(full_array_y[:, idx] for idx, feature in enumerate(FEATURE_NAMES))
        dataset = tf.data.Dataset.from_tensor_slices((x_dict, y_output))

        logger.info("Start shuffling...")
        # Providing input for the model is now handled by Tensorflow since it's maximally optimized
        # Shuffle all samples
        dataset = dataset.shuffle(buffer_size=dataset_size)

        logger.info("Start batching...")
        # Creating new batches of the data
        dataset = dataset.batch(TRAINING_BATCH_SIZE)
        # Automates how TF prefetches the batches for better resource use
        # Since we already tell TF to shuffle all samples and the samples are all stored in the dict in the RAM,
        # this could have no effect at all (maybe on GPU training)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        logger.info("Start training...")

        training_callback = TrainingCallback()

        # verbose set to 0, since we use custom callbacks instead
        history = model.model.fit(dataset, epochs=TRAINING_EPOCHS, verbose=0, callbacks=[training_callback])

        logger.info("Finished training %s", model.model_id)

        if isinstance(history, History):
            plot.plot_training(history, model.model_id)

    except Exception as e:
        raise Exception(f"Training failed: {e}") from e
