import logging

from tensorflow.keras.layers import LSTM, Concatenate, Dense, Dropout, Embedding, Input  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import Adam

from music_generation_lstm.config import LEARNING_RATE, SEQUENCE_LENGTH, TRAINING_ARCHITECTURE

logger = logging.getLogger(__name__)

class BaseModel:
    #   Base model class, that defines an abstract implementation of a model class
    #
    #

    def __init__(self, model_id: str, input_shape: tuple[int, int]):
        self.model_id = model_id
        self.input_shape = input_shape
        self.model: Model

    def build(self):
        #   Should define the architecture of a model
        #
        #

        raise NotImplementedError

    def set_model(self, model: Model):
        self.model = model

    def get_input_shape(self) -> tuple[int, int]:
        return self.input_shape


class LSTMModel(BaseModel):
    #   LSTM model class, that implements the architecture of an lstm model
    #
    #

    def __init__(self, model_id: str, input_shape: tuple[int, int]):
        super().__init__(model_id=model_id, input_shape=input_shape)

    def build(self, vocab_sizes, embedding_dims=32, lstm_units=128):
        #   Builds the architecture of the lstm model
        #
        #

        if TRAINING_ARCHITECTURE == "ADVANCED":
            self.build_model(vocab_sizes=vocab_sizes)
            return
        print("Training using basic architecture")

        # Inputs for each of the 6 features
        input_layers = {name: Input(shape=(SEQUENCE_LENGTH,), name=f"{name}") for name in vocab_sizes}

        # Embeddings for each feature
        embedding_layers = {
            name: Embedding(input_dim=vocab_sizes[name], output_dim=embedding_dims)(input_layers[name])
            for name in vocab_sizes
        }

        # Concatenate embedded features
        x = Concatenate()(list(embedding_layers.values()))  # shape: (batch_size, sequence_length, embedding_dims * 6)

        # LSTM layers
        x = LSTM(lstm_units, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(lstm_units)(x)

        # Output layer: predicting full next event as flattened 6 features
        outputs = []
        for name in vocab_sizes:
            outputs.append(Dense(vocab_sizes[name], activation="softmax", name=f"{name}_output")(x))

        self.model = Model(inputs=list(input_layers.values()), outputs=outputs)
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics={
                "bar_output": "accuracy",
                "position_output": "accuracy",
                "pitch_output": "accuracy",
                "duration_output": "accuracy",
                "velocity_output": "accuracy",
                "tempo_output": "accuracy",
            },
        )

    def build_model(
        self,
        vocab_sizes: dict,
        sequence_length: int = SEQUENCE_LENGTH,  # Default for this architecture -> 32
        embedding_dim: int = 64,
        lstm_units: int = 256,
        num_lstm_layers: int = 3,
        dropout_rate: float = 0.20,
    ):
        print("Training using advanced architecture")

        # Input + Embedding layers (one per feature)
        inputs = {}
        embedded = []
        for feature, vocab_size in vocab_sizes.items():
            inp = Input(shape=(sequence_length,), name=f"{feature}")
            emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name=f"{feature}_emb")(inp)
            inputs[feature] = inp
            embedded.append(emb)

        # Concatenate all embeddings along the feature axis
        x = Concatenate(name="concat_embeddings")(embedded)

        # Stacked LSTM layers with dropout
        for layer_idx in range(num_lstm_layers):
            # return_sequences=True except on the final layer
            return_sequences = layer_idx < num_lstm_layers - 1

            x = LSTM(units=lstm_units, return_sequences=return_sequences, name=f"lstm_{layer_idx + 1}")(x)
            x = Dropout(rate=dropout_rate, name=f"dropout_{layer_idx + 1}")(x)

        # Separate Dense heads for each feature
        outputs = []
        for feature, vocab_size in vocab_sizes.items():
            out = Dense(units=vocab_size, activation="softmax", name=f"{feature}_output")(x)
            outputs.append(out)

        # Compile losses & metrics for each feature
        built_model = Model(inputs=list(inputs.values()), outputs=outputs, name="midi_sixtuple_lstm")
        optimizer_selection = Adam(learning_rate=LEARNING_RATE)
        built_model.compile(
            optimizer=optimizer_selection,
            loss={f"{feature}_out": "sparse_categorical_crossentropy" for feature in vocab_sizes},
            metrics={f"{feature}_out": "accuracy" for feature in vocab_sizes},
        )
        self.model = built_model
