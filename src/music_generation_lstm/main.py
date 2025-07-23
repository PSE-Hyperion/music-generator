import logging

from music_generation_lstm import cli
from music_generation_lstm.logging_config import setup_logging


def main():
    """
    Starts the program by starting logging and a cli session
    """
    setup_logging(level="INFO")
    logging.getLogger(__name__).info("Starting CLI")

    cli.start_session()


if __name__ == "__main__":
    main()


""" Use this for testing the detokenizer
# entry point for  script execution
if __name__ == "__main__":
    import os

    from music_generation_lstm.config import DATASETS_MIDI_DIR
    from music_generation_lstm.midi import parser, writer
    from music_generation_lstm.processing.tokenization import tokenizer as t

    midi_path = "Tokenize Dis/twinkle.mid"
    tokenizer = t.Tokenizer("")
    writer.write_midi(
        "test_twinkle", t.detokenize(tokenizer.tokenize(parser.parse_midi(os.path.join(DATASETS_MIDI_DIR, midi_path))))
    )

    # main()
"""
