# lstm model class, that can be build (architecture), loaded or saved from the data/models folder. Should contain the tf.keras model, token map, metadata
import tensorflow as tf
from keras import models, Sequential, Input

class LSTMModel():
    def __init__(self):
        self.model = models.Model
        self.TYPE = "LSTM"

    # builds the architecture
    def build(self, input_shape : tuple[int, int]):
        self.model = Sequential(
            Input(shape=input_shape),

        )

    # saves the tf.keras model and other information, in a named folder, as a .h5 in the models folder
    def save(self):
        pass

    # loads the tf.keras model and other information
    @classmethod
    def load(cls):
        pass
