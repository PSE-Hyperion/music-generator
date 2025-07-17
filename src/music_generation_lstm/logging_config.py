import logging
import sys

"""
Trying to load packages for colored logging outputs
"""
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    class Fore:
        RED = YELLOW = GREEN = CYAN = RESET = ""
    class Style:
        BRIGHT = NORMAL = RESET_ALL = ""

class ColoredFormatter(logging.Formatter):
    """
    Config for cool looking logs.
    Feel free to play around with it, it's just a template i used.
    """
    def format(self, record):
        name = record.name.upper()
        msg = record.getMessage()

        # Choose color for the levels
        if record.levelno >= logging.ERROR:
            color = Fore.RED + Style.BRIGHT
            prefix = f"[{name} ERROR:]"
        elif record.levelno >= logging.WARNING:
            color = Fore.YELLOW
            prefix = f"[{name}] WARNING:"
        elif record.levelno == logging.INFO:
            color = Fore.CYAN
            prefix = f"[{name}]"
        else:  # DEBUG & others
            color = Fore.GREEN
            prefix = f"[{name}]"

        return f"{color}{prefix} {msg}{Style.RESET_ALL}"

def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root logger with a single console (stderr) handler.

    Args:
        level: The minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to "INFO".
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())

    # Removes any existing handlers: needed for custom logging
    # Replace with defined one
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
