import logging

from tensorflow.keras import callbacks  # type: ignore

logger = logging.getLogger(__name__)


class TrainingCallback(callbacks.Callback):
    """
    Custom callback class to use as callback in model.fit of a keras model.

    Replaces the standard training callbacks, that were hard to read (multiple features, both accuracy and loss)
    with more readable custom version.
    """

    def __init__(self):
        super().__init__()
        self.header_printed = False

    def on_train_begin(self, logs=None) -> None:
        """
        Called once before the first epoch to log training start.
        """
        logger.info("Started training...")

    def on_epoch_end(self, epoch, logs: dict[str, float] | None = None) -> None:
        """
        Automatically gets called in model.fit of keras model during training, at the end of an epoch.

        Prints (logs) the training result of this epoch in a more readable way than standard keras callback.
        """

        logs = logs or {}

        outputs = {key.split("_")[0] for key in logs if key != "loss" and "_" in key}

        col_width = 20
        epoch_width = 7
        loss_width = 10

        header = [f"{'Epoch':<{epoch_width}}", f"{'Loss':<{loss_width}}"]
        for output in outputs:
            label = output.split("_")[0].capitalize()
            header.append(f"{label:<{col_width}}")

        # Build values
        values = [f"{epoch + 1:<{epoch_width}}", f"{logs.get('loss', 0):<{loss_width}.3f}"]
        for output in outputs:
            loss_key = f"{output}_output_loss"
            acc_key = f"{output}_output_accuracy"
            if output == "loss":
                loss_key = "loss"
                acc_key = ""
            loss = logs.get(loss_key, 0.0)
            acc = logs.get(acc_key, 0.0)
            values.append(f"{loss:.3f}, {acc:.3f}".ljust(col_width))

        # print header once on first epoch for visual clarity (makes the output appear like a table)
        if not self.header_printed:
            print(" | ".join(header))
            self.header_printed = True

        print(" | ".join(values))

        """
        EPOCH   |   Loss/Accuracy   |   BAR     |   POSITION    | etc
        0       |   Loss            |   2       |   4           | etc
        0       |   Accuracy        |   0.2     |   0.1         | etc
        1       |   Loss            |   1.5     |   2           |
        1       |   Accuracy        |   0.3     |   0.3         |
        """
