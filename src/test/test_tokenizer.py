import os

from music_generation_lstm.config import DATASETS_MIDI_DIR
from music_generation_lstm.midi import parser, writer
from music_generation_lstm.processing.tokenization import tokenizer as t


def test_tokenize_detokenize(midi_file: str):
    """
    Tokenizes and immediatly detokenizes midi song at given path. Stores result in results data folder
    """

    tokenizer = t.Tokenizer("")
    writer.write_midi(
        "tokenize_detokenize_result",
        t.detokenize(tokenizer.tokenize(parser.parse_midi(os.path.join(DATASETS_MIDI_DIR, midi_file)))),
    )


"""

Für chords und noten gleichzeitig gespielt unterschiedliche länge (ähnliche länge und stark unterschiedliche länge)
1bitdragon files erstellen

Diese detokenize(tokenize(x))

Prüfen, ob mehr chords erstellt werden als sollten

"""
