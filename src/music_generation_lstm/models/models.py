from keras import Sequential
from keras.src.layers import LSTM, Embedding, Concatenate, Dense, Dropout, Input
from keras.src.models import Model
from typing import Optional, Final

from config import SEQUENCE_LENGTH

class BaseModel():
    #
    #   Base model class, that defines an abstract implementation of a model class
    #

    TYPE = ""

    def __init__(self, name : str, input_shape : tuple[int, int]):
        self.name = name
        self.input_shape = input_shape
        self.model: Model

    def build(self):
        raise NotImplementedError



class LSTMModel(BaseModel):
    #
    #   LSTM model class, that implements the architecture of an lstm model
    #

    TYPE: Final = "LSTM"

    def __init__(self, name : str, input_shape : tuple[int, int]):
        super().__init__(name=name, input_shape=input_shape)


    # builds the architecture
    def build(self, vocab_sizes, embedding_dims=32, lstm_units=128):
        # Inputs for each of the 6 features
        input_layers = {
            name: Input(shape=(SEQUENCE_LENGTH,), name=f"{name}")
            for name in vocab_sizes
        }

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
                    'type_output': 'accuracy',
                    'pitch_output': 'accuracy',
                    'duration_output': 'accuracy',
                    'delta_offset_output': 'accuracy',
                    'velocity_output': 'accuracy',
                    'instrument_output': 'accuracy',
                }
            )



class ModelFactory():
    MODEL_MAP = {
        LSTMModel.TYPE : LSTMModel
        #"Transformer.TYPE: TransformerModel()
    }

    @staticmethod
    def get_model_types() -> list[str]:
        model_types = []
        for type in ModelFactory.MODEL_MAP:
            model_types.append(type)
        return model_types

    @staticmethod
    def create_model(model_type : str, model_name : str, input_shape : tuple[int, int]) -> BaseModel | None:    # new stuff learned
        try:
            return ModelFactory.MODEL_MAP[model_type](model_name, input_shape=input_shape)
        except Exception as e:
            print(f"Creating model {model_name} of type {model_type} failed {e}")
            return None

    @staticmethod
    def does_type_exist(model_type : str):
        return ModelFactory.MODEL_MAP.__contains__(model_type)
