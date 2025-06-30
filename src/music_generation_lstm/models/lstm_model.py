# lstm model class, that can be build (architecture), loaded or saved from the data/models folder. Should contain the tf.keras model, token map, metadata

class LSTMModel():
    def __init__(self):
        pass

    # builds the architecture
    def build(self):
        pass

    # saves the tf.keras model and other information, in a named folder, as a .h5 in the models folder
    def save(self):
        pass

    # loads the tf.keras model and other information
    @classmethod
    def load(cls):
        pass