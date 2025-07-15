from tensorflow.keras import callbacks  # type: ignore


class TrainingCallback(callbacks.Callback):
    """
    Custom callback class to use as callback in model.fit of a keras model.

    Replaces the standard training callbacks, that were hard to read (multiple features, both accuracy and loss)
    with more readable custom version.
    """

    def on_epoch_end(self, epoch, logs=None):
        """
        Automatically gets called in model.fit of keras model during training, at the end of an epoch.

        Prints (logs) the training result of this epoch in a more readable way than standard keras callback.
        """

        keys = list(logs.keys())
        watch = keys[0]  ## watching the first key, usually the loss
        print(f"epoch {epoch} {watch}: {logs[watch]:.3f}          ")
