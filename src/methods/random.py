from numpy.random import default_rng

def random_weighting(N, *args, **attributes):
    rng = default_rng()
    weights = rng.uniform(size=len(N))
    weights = weights / sum(weights)
    return weights
