import logging
import time

from tensorflow.keras import callbacks  # type: ignore

logger = logging.getLogger(__name__)


class TrainingCallback(callbacks.Callback):
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
            f"{'Epoch':<{TrainingCallback.EPOCH_COLUMN_WIDTH}}",
            f"{'Time':<{TrainingCallback.TIME_COLUMN_WIDTH}}",
            f"{'Loss':<{TrainingCallback.LOSS_COLUMN_WIDTH}}",
        ]

        for output in outputs:
            label = output.split("_")[0].capitalize()
            header.append(f"{label:<{TrainingCallback.COLUMN_WIDTH}}")

        end_time = time.time()
        duration = end_time - self._start_time
        mins, secs = divmod(duration, 60)

        values = [
            f"{epoch + 1:<{TrainingCallback.EPOCH_COLUMN_WIDTH}}",
            f"{f'{int(mins)}m {int(secs)}s':<{TrainingCallback.TIME_COLUMN_WIDTH}}",
            f"{logs.get('loss', 0):<{TrainingCallback.LOSS_COLUMN_WIDTH}.{TrainingCallback.FLOAT_PRECISION}f}",
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
                f"{loss:.{TrainingCallback.FLOAT_PRECISION}f}, {acc:.{TrainingCallback.FLOAT_PRECISION}f}".ljust(
                    TrainingCallback.COLUMN_WIDTH
                )
            )

        # print header once on first epoch for visual clarity (makes the output appear like a table)
        if not self._header_printed:
            print(" | ".join(header))
            self._header_printed = True

        print(" | ".join(values))
