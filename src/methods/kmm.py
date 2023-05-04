import numpy as np

from utils.metrics import calculate_rbf_gamma
from adapt.instance_based import KMM


def kernel_mean_matching(N, R, columns, *args, **attributes):
    N_columns = N[columns].values
    R_columns = R[columns].values
    sqrt_N = np.sqrt(len(N))
    gamma = calculate_rbf_gamma(np.append(N_columns, R_columns, axis=0))
    epsilon = (sqrt_N - 1) / sqrt_N
    model = KMM(
        kernel="rbf",
        gamma=gamma,
        B=1000,
        verbose=0,
        max_size=len(N),
        max_iter=5000,
        tol=1e-5,
        eps=epsilon
    )
    weights = model.fit_weights(N_columns, R_columns)
    return np.squeeze(weights) / np.sum(weights)
