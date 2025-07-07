from tensorflow.keras.layers import LSTM, Concatenate, Dense, Dropout, Embedding, Input  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

from music_generation_lstm.config import SEQUENCE_LENGTH


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
