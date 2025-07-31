import os

from groove_panda.config import Config
from groove_panda.midi import parser, writer
from groove_panda.processing.tokenization import tokenizer as t

config = Config()


def test_tokenize_detokenize(midi_file: str):
    """
    Tokenizes and immediatly detokenizes given midi song in test folder (assummes folder called test in datasets).

    Stores result in results data folder.
    """

    tokenizer = t.Tokenizer("")
    writer.write_midi(
        "tokenize_detokenize_result",
        t.detokenize(tokenizer.tokenize(parser.parse_midi(os.path.join(config.datasets_midi_dir, "test", midi_file)))),
    )


"""

Für chords und noten gleichzeitig gespielt unterschiedliche länge (ähnliche länge und stark unterschiedliche länge)
1bitdragon files erstellen

Diese detokenize(tokenize(x))

Prüfen, ob mehr chords erstellt werden als sollten

"""
