import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import History, TensorBoard, EarlyStopping  # type: ignore

from groove_panda.config import FEATURE_NAMES, LOG_DIR, TRAINING_EPOCHS, VALIDATION_DATASET_SIZE
from groove_panda.models import plot
from groove_panda.models.flexible_sequence_generator import FlexibleSequenceGenerator
from groove_panda.models.models import BaseModel
from groove_panda.models.tf_custom.callbacks import TerminalPrettyCallback
from groove_panda.processing.process import extract_subsequence

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
            len(train_generator.sample_map),
        )
        logger.info("Sequence length: %s, Stride: %s", train_generator.sequence_length, train_generator.stride)
        logger.info("Steps per epoch: %s, Batch size: %s", steps_per_epoch, train_generator.batch_size)

        training_callback = TerminalPrettyCallback()
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)  # type: ignore

        # fit() will automatically call on_epoch_end of lazy sequence generator, to get new samples
        history = model.model.fit(
            train_generator,
            epochs=TRAINING_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            verbose=2,  # type: ignore
            callbacks=[training_callback, tensorboard_cb],
            # Note: validation_split doesn't work with generators,
            # you'd need a separate validation generator (or other solution)
        )

    except Exception as e:
        raise Exception(f"Training failed: {e}") from e

    logger.info("Finished training %s", model.model_id)

    if isinstance(history, History):
        plot.plot_training(history, model.model_id)


def train_model_eager(model: BaseModel, train_generator: FlexibleSequenceGenerator):
    """
    Loads all necessary data upfront.
    This reduces the latency per sample. The model can take the samples
    directly from the RAM and doesn't have to load a file.
    It leads to better use of the computation units resources.
    It should be the most efficient way of training until the size of the dataset is at least 1 GB
    """

    try:
        logger.info("Start gathering all subsequences from continuous data...")

        # Extract ALL possible subsequences using FlexibleSequenceGenerator logic
        all_x, all_y = [], []

        for continuous_seq in train_generator.song_data:
            max_start_idx = len(continuous_seq) - train_generator.sequence_length - 1
            max_start_steps = max_start_idx // train_generator.stride

            for start_step in range(max_start_steps + 1):
                start_idx = start_step * train_generator.stride

                # Extract subsequence using the same logic as FlexibleSequenceGenerator
                x, y = extract_subsequence(
                    continuous_seq, train_generator.sequence_length, stride=train_generator.stride, start_idx=start_idx
                )
                all_x.append(x)
                all_y.append(y)

        # Convert to numpy arrays
        full_x_array = np.array(all_x)
        full_y_array = np.array(all_y)

        assert full_x_array.shape[0] == full_y_array.shape[0]
        dataset_size = full_x_array.shape[0]
        train_dataset_size = int((1 - VALIDATION_DATASET_SIZE) * dataset_size)

        logger.info("Loaded %d total subsequences from %d songs", dataset_size, len(train_generator.song_data))

        logger.info("Splitting the dataset into training and validation...")

        train_x_array = full_x_array[:train_dataset_size]
        train_y_array = full_y_array[:train_dataset_size]
        val_x_array = full_x_array[train_dataset_size:]
        val_y_array = full_y_array[train_dataset_size:]

        logger.info("Start converting the data into the required format for Keras...")
        # Conversion for the model input layers
        # Iterating over the feature axis of the tensors
        train_x_dict = {
            f"input_{feature}": train_x_array[:, :, idx]  # take of each sample only the specified feature
            for idx, feature in enumerate(FEATURE_NAMES)
        }
        train_y_dict = {
            f"output_{feature}": train_y_array[:, idx]  # take of each sample only the specified feature
            for idx, feature in enumerate(FEATURE_NAMES)
        }
        val_x_dict = {
            f"input_{feature}": val_x_array[:, :, idx]  # take of each sample only the specified feature
            for idx, feature in enumerate(FEATURE_NAMES)
        }
        val_y_dict = {
            f"output_{feature}": val_y_array[:, idx]  # take of each sample only the specified feature
            for idx, feature in enumerate(FEATURE_NAMES)
        }

        logger.info("Giving dataset to TensorFlow...")
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x_dict, train_y_dict))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_x_dict, val_y_dict))

        # Providing input for the model is now handled by Tensorflow since it's maximally optimized
        # The full pipeline will be executed each epoch (tf.data handles the steps as functions, not as the results of the functions)
        #
        # Shuffle all samples for each epoch (only for training dataset because validation should be consistent over epochs)
        # Create new batches with the shuffled samples each epoch
        #
        # Prefetching automates how TF prefetches the batches for better resource use
        # Since we already tell TF to shuffle all samples and the samples are all stored in the dict in the RAM,
        # this could have no effect at all (maybe on GPU training)

        train_dataset = train_dataset.shuffle(buffer_size=dataset_size)

        train_dataset = train_dataset.batch(train_generator.batch_size)
        val_dataset = val_dataset.batch(train_generator.batch_size)

        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        # Callbacks for pretty printing in the terminal and for TensorBoard logging
        # Early stopping ensures that the training stops when the validation loss doesn't improve
        callbacks = [
            TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]

        logger.info("Start training...")

        history = model.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=TRAINING_EPOCHS,
            callbacks=callbacks,
        )

        logger.info("Finished training %s", model.model_id)

        if isinstance(history, History):
            plot.plot_training(history, model.model_id)

    except Exception as e:
        raise Exception(f"Training failed: {e}") from e
