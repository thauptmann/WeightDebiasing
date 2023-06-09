import numpy as np

from utils.metrics import calculate_rbf_gamma
from adapt.instance_based import KMM


def kernel_mean_matching(N, R, columns, *args, **attributes):
    N_columns = N[columns].values
    R_columns = R[columns].values
    gamma = calculate_rbf_gamma(np.append(N_columns, R_columns, axis=0))
    model = KMM(
        kernel="rbf",
        gamma=gamma,
        B=1000,
        verbose=0,
        max_size=len(N),
        max_iter=10000,
        tol=1e-5,
    )
    weights = model.fit_weights(N_columns, R_columns)
    return np.squeeze(weights) / np.sum(weights)
