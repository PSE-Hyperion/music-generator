import logging

from groove_panda import cli
from groove_panda.config import Config
from groove_panda.logging_config import setup_logging


def main():
    """
    Starts the program by starting logging and a cli session
    """

    setup_logging(level="INFO")
    logging.getLogger("main").info("Starting CLI")
    config = Config()
    config.load_config("config")
    cli.start_session()


# entry point for script execution
if __name__ == "__main__":
    main()
