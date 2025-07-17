import logging


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root logger with a single console (stderr) handler.

    Args:
        level: The minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to "INFO".
    """

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Removes any existing handlers: needed for custom logging
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
