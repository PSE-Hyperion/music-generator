import logging

from music_generation_lstm import cli


def main():
    """
       logger.
 Starts the program by starting a cli session
    """
    logging.basicConfig(level=logging.DEBUG)
    cli.start_session()


# entry point for  script execution
if __name__ == "__main__":
    main()
