import logging
import os

from music_generation_lstm import cli
from music_generation_lstm.logging_config import setup_logging


def main():
    """
    Starts the program by starting logging and a cli session
    """
    setup_logging(level="INFO")
    logging.getLogger("main").info("Starting CLI")
    init_envs()

    cli.start_session()

def init_envs():
    """ Setting up environment variables for parts of the program """

    # Log level for compiled operations of TensorFlow
    os.putenv("TF_CPP_MIN_LOG_LEVEL", 2)
    # TensorFlow options for more performance but less reproducibility
    os.putenv("TF_ENABLE_ONEDNN_OPTS", 1)
    os.putenv("TF_USE_CUDNN", 1)


# entry point for  script execution
if __name__ == "__main__":
    main()
