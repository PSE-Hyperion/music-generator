from keras import Sequential
from keras.src.layers import LSTM, Dense, Dropout, Input
from keras.src.models import Model
from typing import Optional, Final

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
        self.build()


    # builds the architecture
    def build(self):
        self.model = Sequential([
            Input(shape=self.input_shape),                      # Input() instead of direct assignment
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(128),
            Dense(units=self.input_shape[1], activation="softmax")       # not sure if input_shape[1] (raises error, but previous solution) or input_shape
        ])
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
        print(f"Model {self.name} build")


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
