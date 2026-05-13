"""
Colour-coded console logger
"""
import logging, sys

class _ColourFmt(logging.Formatter):
    CODES = {
        logging.DEBUG:    "\033[36m",
        logging.INFO:     "\033[32m",
        logging.WARNING:  "\033[33m",
        logging.ERROR:    "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, r):
        r.levelname = f"{self.CODES.get(r.levelno,'')}{r.levelname:<8}{self.RESET}"
        return super().format(r)

def get_logger(name: str = "credit") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(_ColourFmt(
            fmt="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log
