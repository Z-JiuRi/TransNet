import logging
import os
import sys

line_seg = ''.join(['*'] * 10)
logger = logging.getLogger("transnet")


def setup_logging(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(levelname).1s %(asctime)s %(filename)s:%(lineno)-4d] %(message)s',
        datefmt='%m.%d/%H:%M:%S')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(exp_dir, "run.log"), mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
