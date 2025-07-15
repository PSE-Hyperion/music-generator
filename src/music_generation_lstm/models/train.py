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
