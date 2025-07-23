import os

from groove_panda import config
from groove_panda.config import DATASETS_MIDI_DIR, Parser, TokenizeMode
from groove_panda.midi import parser, writer
from groove_panda.processing.tokenization import tokenizer as t


def test_tokenize_detokenize(midi_file: str):
    """
    Tokenizes and immediatly detokenizes given midi song in test folder (assummes folder called test in datasets).

    Stores result in results data folder.

    Uses mido as parser and tokenizer mode original.
    """
    config.PARSER = Parser.MIDO
    config.TOKENIZE_MODE = TokenizeMode.ORIGINAL
    tokenizer = t.Tokenizer("")
    writer.write_midi(
        "tokenize_detokenize_mido_result",
        t.detokenize(tokenizer.tokenize(parser.parse_midi(os.path.join(DATASETS_MIDI_DIR, "test", midi_file)))),
    )
