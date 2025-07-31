import logging

import tensorflow as tf
from tensorflow.keras.callbacks import History  # type: ignore
from tensorflow.keras.layers import LSTM, Concatenate, Dense, Dropout, Embedding, Input  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from groove_panda.config import MODEL_PRESETS
from groove_panda.models.tf_custom.callbacks import TerminalPrettyCallback
from groove_panda.models.tf_custom.losses import SoftCategoricalKLDivergence
from groove_panda.models.utils import get_loss_weights

logger = logging.getLogger(__name__)


class BaseModel:
    #   Base model class, that defines an abstract implementation of a model class
    #
    #

    def __init__(self, model_id: str, input_shape: tuple[int, int]):
        self._model_id = model_id
        self._input_shape = input_shape
        self._model: Model
        self._epochs_trained: int = 0
        self._history: History = History()  # Initialises an empty History attribute, populated after training
        self._version: int = 1

    def build(self):
        #   Should define the architecture of a model
        #
        #

        raise NotImplementedError

    def train(self, dataset, epochs, callbacks, tensorboard):
        raise NotImplementedError

    def set_model(self, model: Model):
        self._model = model

    def setup(self, history: History, version: int, epochs_trained: int):
        """
        Sets this model's internal configurations to match those it had after being last trained.
        This includes its history as well as the epochs the model has previously been trained on.
        Also increments the model's version counter.
        """
        self._history = history
        self._version = version + 1
        self.add_epochs(epochs_trained)

    def set_history(self, history: History):
        """
        Sets this model's history to the given argument, including the epochs
        this model was trained on in previous sessions.
        """
        self._history = history

    def add_epochs(self, new_epochs: int):
        self._epochs_trained += new_epochs

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def input_shape(self) -> tuple[int, int]:
        return self._input_shape

    @property
    def model(self) -> Model:
        return self._model

    @property
    def epochs_trained(self) -> int:
        return self._epochs_trained

    @property
    def history(self) -> History:
        return self._history

    @property
    def version(self) -> int:
        return self._version


class LSTMModel(BaseModel):
    #
    # LSTM model class, that implements the architecture of an LSTM model
    #

    def __init__(self, model_id: str, input_shape: tuple[int, int]):
        super().__init__(model_id=model_id, input_shape=input_shape)

    def build(self, vocab_sizes: dict[str, int], preset_name: str = "basic"):
        """
        Builds the LSTMModel according to the hyperparameters defined
        in MODEL_PRESETS[preset_name].

        Arguments:
            vocab_sizes: A dict mapping each feature name to its vocabulary size.
            preset_name: The key for the preset in MODEL_PRESETS to use.
        """

        if preset_name not in MODEL_PRESETS:
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {list(MODEL_PRESETS.keys())}")

        logger.info(f"Training new model with the '{preset_name}' architecture preset.")

        # Load configuration
        config = MODEL_PRESETS[preset_name]

        sequence_length: int = config["sequence_length"]
        lstm_units: int = config["lstm_units"]
        num_lstm_layers: int = config["num_lstm_layers"]
        dropout_rate: float = config["dropout_rate"]
        learning_rate: float = config["learning_rate"]
        embedding_dims_config = config["embedding_dims"]  # Raw embedding dims, either an int or a dict

        if isinstance(embedding_dims_config, int):
            # If user gave a single integer, apply it to all features
            embedding_dims: dict[str, int] = dict.fromkeys(vocab_sizes, embedding_dims_config)
        else:
            embedding_dims: dict[str, int] = embedding_dims_config

        # Create one Input() per feature, each taking a sequence of tokens
        input_layers = {
            feature_name: Input(
                shape=(sequence_length,),  # -> Ex: (32,)
                name=f"input_{feature_name}",
            )
            for feature_name in vocab_sizes
        }

        # Wrap each Input in an Embedding layer using each input's dimension
        embedding_layers = {}
        for feature_name, vocab_size in vocab_sizes.items():
            # Look up how many dimensions we want for this feature
            feature_embedding_dim = embedding_dims[feature_name]

            # Build the Embedding layer
            embedded_tensor = Embedding(
                input_dim=vocab_size,  # Size of this feature's vocabulary.
                # Embedding matrix has [vocab_size] rows to choose from for each feature
                output_dim=feature_embedding_dim,  # Ex: Pitch -> 128, Bar -> 8, etc.
                name=f"embedding_{feature_name}",  # Helps with debugging & saving
            )(input_layers[feature_name])  # Apply embedding to the corresponding Input()
            embedding_layers[feature_name] = embedded_tensor

        # Concatenate all feature embeddings into one vector
        # Resulting shape: (batch, sequence_length, sum_of_all_embedding_dims)
        x = Concatenate(name="concat_all_embeddings")(
            list(embedding_layers.values())  # Creates a list of all the embedded_tensors found in the dict.
        )

        for layer_index in range(num_lstm_layers):
            # Decide whether this LSTM layer should return the full sequence
            # (needed by the next LSTM layer) or just the last output
            is_last_layer = layer_index == num_lstm_layers - 1
            return_sequences_flag = not is_last_layer

            # Build the LSTM layer
            # - units: how many hidden units in this layer
            # - return_sequences: True for all but the last layer, so layers have temporal knowledge
            x = LSTM(units=lstm_units, return_sequences=return_sequences_flag, name=f"lstm_layer_{layer_index + 1}")(x)

            # Add a Dropout layer to avoid overfitting.
            x = Dropout(rate=dropout_rate, name=f"dropout_after_lstm_{layer_index + 1}")(x)

        # Add a per feature dense layer
        output_tensors = {}
        for feature_name, vocab_size in vocab_sizes.items():
            # Build a Dense softmax head for this feature
            dense_output = Dense(
                units=vocab_size,  # Number of classes for this feature
                activation="softmax",  # We want a probability distribution
                name=f"output_{feature_name}",
            )(x)  # Append this dense layer to the last LSTM/Dropout output
            output_tensors[f"output_{feature_name}"] = dense_output

        # Create the model object
        built_model = Model(
            inputs=list(input_layers.values()),
            outputs=output_tensors,  # Should it not be list(output_tensors.values())?????????
            name=f"{self.model_id}_midi_lstm",
        )

        # Prepare and set the loss function and metrics for each output
        loss_dict = {
            f"output_{feature_name}": SoftCategoricalKLDivergence([0.05, 0.1, 0.7, 0.1, 0.05])
            for feature_name in vocab_sizes
        }
        metric_dict = {f"output_{feature_name}": "accuracy" for feature_name in vocab_sizes}

        # Compile model using the specified learning rate
        # Adding gradient clipping to avoid extreme gradient values that may destroy the learning process
        optimizer = Adam(
            learning_rate=learning_rate,
            clipnorm=1.0
        )
        built_model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            loss_weights=get_loss_weights(),
            metrics=metric_dict
        )

        # Assign model to this Model object's LSTM model.
        self._model = built_model

    def train(self, dataset: tf.data.Dataset, epochs: int, callbacks: TerminalPrettyCallback, tensorboard):
        """
        Trains the model using the provided sequence generator for the provided number of epochs.
        Updates the model's history to reflect data from ALL training sessions.
        """
        # Train the model for the specified number of epochs
        try:
            # Verbose set to 0, since we use custom callbacks instead
            total_epochs = self._epochs_trained + epochs
            history = self._model.fit(
                dataset,
                epochs=total_epochs,
                initial_epoch=self._epochs_trained,
                verbose=0,
                callbacks=[callbacks, tensorboard],
            )
            self.add_epochs(epochs)

        except Exception as e:
            raise Exception(f"Training failed: {e}") from e

        # Rebuild history (connecting it to previous histories)
        new_history = history

        if self._history.history:  # If this model has been trained before, combine the histories.
            combined_history = {
                key: self._history.history[key] + new_history.history[key] for key in new_history.history
            }
            new_history.history = combined_history

        self.set_history(new_history)

        return new_history


MODEL_TYPES = {"LSTM": LSTMModel}
