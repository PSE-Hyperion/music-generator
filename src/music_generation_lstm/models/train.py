from typing import Generator

from numpy.typing import NDArray
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import History  # type: ignore

from music_generation_lstm.config import TRAINING_BATCH_SIZE, TRAINING_EPOCHS
from music_generation_lstm.models import plot
from music_generation_lstm.models.lazy_sequence_generator import LazySequenceGenerator
from music_generation_lstm.models.models import BaseModel


def temperature():
    pass


def split_X_y(X, y):
    #
    #   Splits X and y into dictionaries of feature-wise arrays for model input and output
    #

    feature_names = ["type", "pitch", "duration", "delta_offset", "velocity", "instrument"]

    X_dict = {feature_names[i]: X[:, :, i] for i in range(6)}

    y_array = np.array(y, dtype=np.int32)
    y_dict = {feature_names[i]: y_array[:, i] for i in range(6)}

    return X_dict, y_dict


def train_model_without_lazy(model: BaseModel, X, y):
    #   trains the given model
    #
    #

    print(f"Start training {model.model_id}...")
    try:
        X_dict, y_dict = split_X_y(X, y)

        history = model.model.fit(
            X_dict,
            [y_dict[name] for name in ["type", "pitch", "duration", "delta_offset", "velocity", "instrument"]],
            epochs=TRAINING_EPOCHS,
            batch_size=TRAINING_BATCH_SIZE,
            validation_split=0.1,
            verbose=2,  # type: ignore[arg-type]
            # try out 1 and 2. Terminal output is weird with all the different
        )

    except Exception as e:
        raise Exception(f"Training failed {model.model_id} {e}") from train_model_without_lazy
    #    if isinstance(history, History):
    #        plot.plot_training(history, model.model_id)

    return history


def train_model(model: BaseModel, file_paths: list):
    """
    Train model using LazySequenceGenerator for memory-efficient training on large datasets.

    File paths contain the paths to .npz files. Needed to create a lazy sequence generator, that will lazily load
    samples in batches (and random samples and shuffles)
    """

    print(f"Start training {model.model_id} with lazy loading...")

    try:
        train_generator = LazySequenceGenerator(file_paths=file_paths, batch_size=TRAINING_BATCH_SIZE, shuffle=True)

        steps_per_epoch = len(train_generator)

        print(f"Training with {len(file_paths)} files, containing {train_generator.n_samples} total samples")
        print(f"Steps per epoch: {steps_per_epoch}, Batch size: {TRAINING_BATCH_SIZE}")

        # fit() will automatically call on_epoch_end of lazy sequence generator, to get new samples
        history = model.model.fit(
            train_generator,
            epochs=TRAINING_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            verbose=2,  # type: ignore
            # Note: validation_split doesn't work with generators,
            # you'd need a separate validation generator (or other solution)
        )

    except Exception as e:
        raise Exception(f"Training failed: {e}") from train_model

    print(f"Finished training {model.model_id}")

    if isinstance(history, History):
        plot.plot_training(history, model.model_id)


def train_model_sliding(model: BaseModel, *dataset_parts: NDArray):
    data, starts = dataset_parts[0:2]
    training_data_gen = sliding_window_generator(data, starts, 10)
    training_data_signature = {
        'X': tf.TensorSpec(shape=(10, 6), dtype=tf.uint8), # window size 10, feature dim 6
        'y': tf.TensorSpec(shape=(1,6), dtype=tf.uint8)
    }

    dataset = tf.data.Dataset.from_generator(
        generator = (lambda: training_data_gen),
        output_signature = training_data_signature
       )
    dataset = (
        dataset
        .cache()
        .shuffle(10000)
        .batch(64)
        .prefetch(tf.data.AUTOTUNE)
    )
    model.model.fit(dataset, epochs=1)


def sliding_window_generator(sequences: NDArray, sequence_start_indexes: NDArray[int], window_size: int, target_size = 1, stride = 1):
    # 'data' is a concatenated array, the indexes store the start and end indexes of the separate sequences
    # Asserting that sequences are densely packed and the start indexes are matching the sequences
    assert sequences.shape[0] == sequence_start_indexes[-1] - sequence_start_indexes[0]

    max_sequences = sequence_start_indexes.shape[0] - 1
    window_with_target_size = window_size + target_size

    sequence_number = 0
    window_start_idx = sequence_start_indexes[sequence_number]
    window_with_target_end_idx = window_start_idx + window_with_target_size

    while sequence_number < max_sequences:
        if window_with_target_end_idx >= sequence_start_indexes[sequence_number + 1]:
            sequence_number += 1
            window_start_idx = sequence_start_indexes[sequence_number]
            window_with_target_end_idx = window_start_idx + window_with_target_size
            continue

        window_end_idx = window_start_idx + window_size
        X = sequences[window_start_idx:window_end_idx]
        y = sequences[window_end_idx:window_with_target_end_idx]
        yield {'X': X, 'y': y}
        window_start_idx += stride
        window_with_target_end_idx += stride

