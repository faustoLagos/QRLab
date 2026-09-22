import argparse
import numpy as np


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    if value.lower() in ('yes', 'true', 'y', 't', '1'):
        return True

    if value.lower() in ('no', 'false', 'n', 'f', '0'):
        return False

    raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")


def in_degrees(angles):
    return list(map(lambda angle: angle * 180 / np.pi, angles))
