import logging, sys, pathlib

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s — %(levelname)s — %(name)s — %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger
