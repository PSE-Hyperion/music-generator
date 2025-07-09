from functools import reduce

from music_generation_lstm import strategies as strats


def compose(*functions):
    """ Composes functions into a single function """
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


class ProcessingPipeline:
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target
        self.pipeline = compose(
            strats.loader('lazy-midi-paths'),
            strats.parser('m21'),
            strats.tokenizer('sixtuple'),
            )

    def run(self):
        return self.pipeline(self.source)

