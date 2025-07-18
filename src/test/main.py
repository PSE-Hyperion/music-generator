import logging

from groove_panda.logging_config import setup_logging
from test.test_tokenizer import test_tokenize_detokenize


def main():
    """
    Starts a test (test method must be called here)
    """

    setup_logging(level="INFO")
    logging.getLogger(__name__).info("Starting test")

    test_tokenize_detokenize("CHORDS_AND_SIMULTANIOUS_NOTES.mid")


# entry point for script execution
if __name__ == "__main__":
    main()
