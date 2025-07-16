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


# entry point for script execution
if __name__ == "__main__":
    main()
