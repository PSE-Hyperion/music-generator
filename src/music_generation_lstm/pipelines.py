from functools import reduce

from music_generation_lstm import strategies as strats


def compose(*functions):
    """ Composes functions into a single function """
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


class ProcessingPipeline:
    def __init__(self, source: str, target: str):
        self.pipeline = compose(
            strats.loader('lazy-midi-paths', source),
            strats.parser('m21'),
            strats.tokenizer('sixtuple'),
            strats.encoder('index-sixtupe'),
            strats.trainig_preparation('none'),
            strats.saver('training-sequences', target)
            )

    def run(self):
        self.pipeline(None)

