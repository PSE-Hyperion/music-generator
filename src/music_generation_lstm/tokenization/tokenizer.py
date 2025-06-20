# tokenizer class, that holds token to int map and can en- and decode token or integer lists

class Tokenizer():
    def __init__(self):
        pass

    # builds the map
    def build(self):
        pass

    # uses map on token list
    def encode(self):
        pass

    # uses reversed map on integer list
    def decode(self):
        pass

    # save map to path (should be saved inside of the corresopnding model)
    def save(self):
        pass

    # load map from path (replaces build call)
    @classmethod
    def load(cls):
        pass

    