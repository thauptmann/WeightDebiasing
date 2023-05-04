import torch
import torch.nn as nn
import ot
from sklearn.metrics.pairwise import rbf_kernel

euclidean_distance_fn = torch.nn.PairwiseDistance(2)


# Weighted MMD loss
class WeightedMMDLoss(nn.Module):
    def __init__(self, gamma, N, R, device):
        super(WeightedMMDLoss, self).__init__()
        self.weights_R = (torch.ones(len(R), dtype=torch.float64) / len(R)).to(device)
        self.weights_R = self.weights_R / torch.sum(self.weights_R)
        self.n_n_rbf_matrix = torch.FloatTensor(rbf_kernel(N, N, gamma)).to(device)
        self.n_r_rbf_matrix = torch.FloatTensor(rbf_kernel(N, R, gamma)).to(device)
        self.gamma = gamma

        r_r_rbf_matrix = torch.matmul(
            torch.unsqueeze(self.weights_R, 1), torch.unsqueeze(self.weights_R, 0)
        ) * torch.FloatTensor(rbf_kernel(R, R, gamma)).to(device)
        self.r_r_mean = r_r_rbf_matrix.sum()

    def forward(self, weights):
        n_n_mean = (
            torch.matmul(torch.unsqueeze(weights, 1), torch.unsqueeze(weights, 0))
            * self.n_n_rbf_matrix
        ).sum()
        n_r_mean = (
            torch.matmul(
                torch.unsqueeze(weights, 1), torch.unsqueeze(self.weights_R, 0)
            )
            * self.n_r_rbf_matrix
        ).sum()

        return torch.sqrt(n_n_mean + self.r_r_mean - 2 * n_r_mean)


class AsamLoss(nn.Module):
    def __init__(self):
        super(AsamLoss, self).__init__()

    def forward(self, N, R, weights):
        weighted_N = torch.mul(N, weights)
        weighted_means_non_representative = torch.sum(weighted_N, axis=0)
        weighted_variance = torch.mul(
            ((N - weighted_means_non_representative) ** 2), weights
        )
        weighted_variance_non_representative = torch.sum(weighted_variance, axis=0)

        means_representative = torch.mean(R, axis=0)
        variance_representative = torch.var(R, axis=0)

        means_difference = means_representative - weighted_means_non_representative

        middle_variance = torch.sqrt(
            (variance_representative + weighted_variance_non_representative) / 2
        )
        loss = torch.abs(means_difference / middle_variance)
        return torch.mean(loss)


# Wasserstein loss
class WassersteinLoss(nn.Module):
    def __init__(self, R, device):
        super(WassersteinLoss, self).__init__()
        len_R = len(R)
        self.weights_R = (torch.ones(len_R, dtype=torch.float64) / len_R).to(device)
        self.device = device

    def forward(self, N, R, weights):
        M = ot.dist(N, R)
        return ot.sinkhorn2(weights, self.weights_R, M, reg=1)


class WeightedMMDWassersteinLoss(nn.Module):
    def __init__(self, gamma, R, device):
        self.mmd_loss = WeightedMMDLoss(gamma, R)
        self.wasserstein = WassersteinLoss(R, device)

    def forward(self, N, R, weights):
        mmd_loss = self.mmd_loss(N, R, weights)
        wasserstein_loss = self.wasserstein(N, R, weights)
        return mmd_loss + wasserstein_loss
