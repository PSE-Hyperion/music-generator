import logging
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, History, TensorBoard  # type: ignore
from tensorflow.keras.utils import Sequence  # type: ignore

from groove_panda import directories
from groove_panda.config import (
    Config,
)
from groove_panda.models import plot
from groove_panda.models.flexible_sequence_generator import FlexibleSequenceGenerator
from groove_panda.models.models import BaseModel
from groove_panda.models.tf_custom.callbacks import TerminalPrettyCallback
from groove_panda.processing.process import extract_subsequence

config = Config()
logger = logging.getLogger(__name__)


def train_model(model: BaseModel, train_generator: Sequence):
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
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=config.log_dir, histogram_freq=1)  # type: ignore

        # fit() will automatically call on_epoch_end of lazy sequence generator, to get new samples
        history = model.model.fit(
            train_generator,
            epochs=config.training_epochs,
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


def train_model_eager(model: BaseModel, train_generator: FlexibleSequenceGenerator) -> None:
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

        # Shuffle all songs to get a less biased split into training and validation dataset.
        # Otherwise the last songs in the list would always be only for validation.
        song_data = train_generator.song_data
        random.Random(config.training_validation_split_seed).shuffle(song_data)

        for continuous_seq in song_data:
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
        train_dataset_size = int((1 - config.validation_split_proportion) * dataset_size)

        song_word = "song" if len(train_generator.song_data) == 1 else "songs"
        logger.info("Loaded %d total subsequences from %d %s", dataset_size, len(train_generator.song_data), song_word)

        logger.info("Splitting the dataset into training and validation...")

        train_x_array = full_x_array[:train_dataset_size].astype(np.int32)
        train_y_array = full_y_array[:train_dataset_size].astype(np.int32)
        val_x_array = full_x_array[train_dataset_size:].astype(np.int32)
        val_y_array = full_y_array[train_dataset_size:].astype(np.int32)

        logger.info("Start converting the data into the required format for Keras...")
        # Conversion for the model input layers
        # Iterating over the feature axis of the tensors

        train_x_dict = {
            f"input_{feature.name}": train_x_array[:, :, idx]  # take of each sample only the specified feature
            for idx, feature in enumerate(config.features)
        }
        train_y_dict = {
            f"output_{feature.name}": train_y_array[:, idx]  # take of each sample only the specified feature
            for idx, feature in enumerate(config.features)
        }
        val_x_dict = {
            f"input_{feature.name}": val_x_array[:, :, idx]  # take of each sample only the specified feature
            for idx, feature in enumerate(config.features)
        }
        val_y_dict = {
            f"output_{feature.name}": val_y_array[:, idx]  # take of each sample only the specified feature
            for idx, feature in enumerate(config.features)
        }

        logger.info("Giving dataset to TensorFlow...")
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x_dict, train_y_dict))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_x_dict, val_y_dict))

        # Providing input for the model is now handled by Tensorflow since it's maximally optimized
        # The full pipeline will be executed each epoch (tf.data handles the steps as functions,
        # not as the results of the functions)
        #
        # Shuffle all samples for each epoch
        # (only for training dataset because validation should be consistent over epochs)
        # Create new batches with the shuffled samples each epoch
        #
        # Prefetching automates how TF prefetches the batches for better resource use
        # Since we already tell TF to shuffle all samples and the samples are all stored in the dict in the RAM,
        # this could have no effect at all (maybe on GPU training)

        if config.enable_training_augmentation:
            train_dataset.map(map_func=augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size=dataset_size, seed=config.dataset_shuffle_seed)

        train_dataset = train_dataset.batch(train_generator.batch_size)
        val_dataset = val_dataset.batch(train_generator.batch_size)

        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        # Callbacks for pretty printing in the terminal and for TensorBoard logging
        # Early stopping ensures that the training stops when the validation loss doesn't improve
        callbacks = [TensorBoard(log_dir=directories.LOG_DIR, histogram_freq=1)]
        if config.early_stopping_enabled:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=config.early_stopping_epochs_to_wait,
                    min_delta=config.early_stopping_threshold,
                    restore_best_weights=True,
                )
            )

        logger.info("Start training...")

        # training_callback = TerminalPrettyCallback() - left out for nowte

        # Other callbacks can be added here for specific purposes

        history = model.train(
            train_dataset,
            val_dataset,
            epochs=config.training_epochs,
            callbacks=callbacks,
        )

        logger.info("Finished training %s", model.model_id)

        if isinstance(history, History):
            plot.plot_training(history, model.model_id)

    except Exception as e:
        raise Exception(f"Training failed: {e}") from e

def augment(x_dict, y_dict):
    x_aug_dict = {}
    y_aug_dict = {}

    for feature in config.features:
        input_feature_name = f"input_{feature.name}"
        output_feature_name = f"output_{feature.name}"

        x_sequence = x_dict[input_feature_name]
        y_value = y_dict[output_feature_name]
        x_aug_sequence = x_sequence
        y_aug_value = y_value

        min_range = feature.min_value
        max_range = feature.max_value

        if feature.name == "pitch":
            (x_aug_sequence, y_aug_value) = augment_pitch(
                x_sequence,
                y_value,
                min_range,
                max_range
            )
        elif feature.name == "velocity":
            (x_aug_sequence, y_aug_value) = augment_velocity(
                x_sequence,
                y_value,
                min_range,
                max_range
            )
        elif feature.name == "tempo":
            (x_aug_sequence, y_aug_value) = augment_tempo(
                x_sequence,
                y_value,
                min_range,
                max_range
            )

        x_aug_dict[input_feature_name] = x_aug_sequence
        y_aug_dict[output_feature_name] = y_aug_value

    return x_aug_dict, y_aug_dict

def augment_pitch(x_sequence, y_value, min_range, max_range):
    global_shift = tf.random.uniform(shape=[],minval=-6,maxval=6,dtype=tf.int32)
    return apply_global_shift(x_sequence, y_value, global_shift, min_range, max_range)

def augment_velocity(x_sequence, y_value, min_range, max_range):
    global_shift = tf.random.uniform(shape=[],minval=-5,maxval=6,dtype=tf.int32)
    return apply_global_shift(x_sequence, y_value, global_shift, min_range, max_range)

def augment_tempo(x_sequence, y_value, min_range, max_range):
    global_shift = tf.random.uniform(shape=[],minval=-2,maxval=3,dtype=tf.int32)
    return apply_global_shift(x_sequence, y_value, global_shift, min_range, max_range)

def apply_global_shift(x_sequence, y_value, shift, min_range, max_range):
    x_shifted = x_sequence + shift
    y_shifted = y_value + shift
    x_shifted = tf.clip_by_value(x_shifted, min_range, max_range)
    y_shifted = tf.clip_by_value(y_shifted, min_range, max_range)
    return (x_shifted, y_shifted)
