import numpy as np


def naive_weighting(N, *args, **attributes):
    weights = np.ones(len(N)) / len(N)
    return weights
