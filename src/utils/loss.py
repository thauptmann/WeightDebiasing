import torch
import torch.nn as nn
import ot

euclidean_distance_fn = torch.nn.PairwiseDistance(2)


# Weighted MMD loss
class WeightedMMDLoss(nn.Module):
    def __init__(self, gamma, N, R, device):
        super(WeightedMMDLoss, self).__init__()
        self.weights_R = (torch.ones(len(R), dtype=torch.double) / len(R)).to(device)
        self.n_n_rbf_matrix = self.rbf_kernel(N, N, gamma)
        self.n_r_rbf_matrix = self.rbf_kernel(N, R, gamma)
        self.gamma = gamma
        self.R = R

        r_r_rbf_matrix = torch.matmul(
            torch.unsqueeze(self.weights_R, 1), torch.unsqueeze(self.weights_R, 0)
        ) * self.rbf_kernel(R, R, gamma)
        self.r_r_mean = r_r_rbf_matrix.sum().to(device)

    def rbf_kernel(self, source, target, gamma):
        distance_matrix = torch.cdist(source, target, p=2)
        squared_distance_matrix = distance_matrix.pow(2)
        return torch.exp(-gamma * squared_distance_matrix)

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
        self.mmd_loss = WeightedMMDLoss(gamma, R, device)
        self.wasserstein = WassersteinLoss(R, device)

    def forward(self, N, R, weights):
        mmd_loss = self.mmd_loss(N, R, weights)
        wasserstein_loss = self.wasserstein(N, R, weights)
        return mmd_loss + wasserstein_loss
