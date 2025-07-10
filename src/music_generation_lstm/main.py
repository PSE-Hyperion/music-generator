import logging

from music_generation_lstm import cli
from music_generation_lstm.logging_config import setup_logger


def main():
    """
    Starts the program by starting a cli session
    """
    setup_logger(logging.DEBUG)
    cli.start_session()


# entry point for  script execution
if __name__ == "__main__":
    main()
