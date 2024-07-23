import numpy as np


def log_transform(x):
    return np.log(np.abs(x.flatten()) + 1)
