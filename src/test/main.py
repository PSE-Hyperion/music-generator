import logging

from groove_panda.logging_config import setup_logging
from test.test_mido_tokenizer import test_tokenize_detokenize


def main():
    """
    Starts a test (test method must be called here)
    """

    setup_logging(level="INFO")
    logging.getLogger(__name__).info("Starting test")

    test_tokenize_detokenize("111.mid")

    logging.getLogger(__name__).info("Ending test")


# entry point for script execution
if __name__ == "__main__":
    main()
