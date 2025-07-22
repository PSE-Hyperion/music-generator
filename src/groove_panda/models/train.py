import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import History  # type: ignore

from groove_panda.config import FEATURE_NAMES, LOG_DIR, TRAINING_BATCH_SIZE, TRAINING_EPOCHS
from groove_panda.models import plot
from groove_panda.models.models import BaseModel
from groove_panda.models.tf_custom.callbacks import TerminalPrettyCallback

logger = logging.getLogger(__name__)


def train_model(model: BaseModel, train_generator):
    """
    Train model using sequence generator for memory-efficient training on large datasets.
    Supports both LazySequenceGenerator (legacy) and FlexibleSequenceGenerator (new).
    """

    logger.info("Start training with sequence generator...")

    try:
        steps_per_epoch = len(train_generator)

        logger.info(
            "Training with %s files, containing %s total samples",
            len(train_generator.file_paths),
            train_generator.total_samples,
        )
        logger.info("Sequence length: %s, Stride: %s", train_generator.sequence_length, train_generator.stride)
        logger.info("Steps per epoch: %s, Batch size: %s", steps_per_epoch, train_generator.batch_size)

        # fit() will automatically call on_epoch_end of lazy sequence generator, to get new samples
        history = model.model.fit(
            train_generator,
            epochs=TRAINING_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            verbose=2,  # type: ignore
            callbacks=[TerminalPrettyCallback()],
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
            f"input_{feature}": full_array_x[:, :, idx]  # take of each sample only the specified feature
            for idx, feature in enumerate(FEATURE_NAMES)
        }
        y_dict = {
            f"output_{feature}": full_array_y[:, idx]  # take of each sample only the specified feature
            for idx, feature in enumerate(FEATURE_NAMES)
        }

        dataset = tf.data.Dataset.from_tensor_slices((x_dict, y_dict))

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

        training_callback = TerminalPrettyCallback()

        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)  # type: ignore
        # Other callbacks can be added here for specific purposes

        # verbose set to 0, since we use cuembeddingstom callbacks instead
        history = model.model.fit(
            dataset, epochs=TRAINING_EPOCHS, verbose=0, callbacks=[training_callback, tensorboard_cb]
        )

        logger.info("Finished training %s", model.model_id)

        if isinstance(history, History):
            plot.plot_training(history, model.model_id)

    except Exception as e:
        raise Exception(f"Training failed: {e}") from e
