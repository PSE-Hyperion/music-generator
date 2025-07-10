import logging
import sys

"""
Config for cool looking logs.
Feel free to play around with it, it's just a template i used.
Maybe move this with other config file(s) into a config package? #REVIEW
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
    def format(self, record):
        name = record.name.upper()
        msg = record.getMessage()

        # Wähle Farbe basierend auf Level
        if record.levelno >= logging.ERROR:
            color = Fore.RED + Style.BRIGHT
            prefix = f"[{name}: ERROR]"
        elif record.levelno >= logging.WARNING:
            color = Fore.YELLOW
            prefix = f"[{name}: WARNING]"
        elif record.levelno == logging.INFO:
            color = Fore.CYAN
            prefix = f"[{name}]"
        else:  # DEBUG & andere
            color = Fore.GREEN
            prefix = f"[{name}]"

        return f"{color}{prefix} {msg}{Style.RESET_ALL}"

def setup_logger(level=logging.DEBUG):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
