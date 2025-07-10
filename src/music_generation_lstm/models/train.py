import numpy as np
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
