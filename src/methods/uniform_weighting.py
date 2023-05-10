import numpy as np


def uniform_weighting(N, *args, **attributes):
    weights = np.ones(len(N)) / len(N)
    return weights / np.sum(weights)
