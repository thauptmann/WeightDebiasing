import torch
import torch.nn as nn
import ot

euclidean_distance_fn = torch.nn.PairwiseDistance(2)


# Weighted MMD loss
class WeightedMMDLoss(nn.Module):
    def __init__(self, gamma, R, device, kernel="rbf"):
        super(WeightedMMDLoss, self).__init__()
        self.gamma = gamma
        self.kernel = self.rbf_kernel if kernel == "rbf" else self.linear_kernel
        len_R = len(R)
        self.weights_R = (torch.ones(len_R, dtype=torch.double) / len_R).to(device)

        y_y_rbf_matrix = torch.matmul(
            torch.unsqueeze(self.weights_R, 1), torch.unsqueeze(self.weights_R, 0)
        ) * self.kernel(R, R, gamma)
        self.y_y_mean = y_y_rbf_matrix.sum().to(device)

    def rbf_kernel(self, source, target, gamma):
        distance_matrix = torch.cdist(source, target, p=2)
        squared_distance_matrix = distance_matrix.pow(2)
        return torch.exp(-gamma * squared_distance_matrix)

    def linear_kernel(self, source, target, gamma):
        dot_product_matrix = torch.mm(source, target.T)
        return dot_product_matrix

    def forward(self, N, R, weights=None):
        if weights is None:
            weights = (torch.ones(len(N), dtype=torch.double) / len(N)).to(self.N)

        x_x_rbf_matrix = torch.matmul(
            torch.unsqueeze(weights, 1), torch.unsqueeze(weights, 0)
        ) * self.kernel(N, N, self.gamma)
        x_x_mean = x_x_rbf_matrix.sum()

        weight_matrix = torch.matmul(
            torch.unsqueeze(weights, 1), torch.unsqueeze(self.weights_R, 0)
        )
        x_y_rbf_matrix = weight_matrix * self.kernel(N, R, self.gamma)
        x_y_mean = x_y_rbf_matrix.sum()

        maximum_mean_discrepancy_value = x_x_mean + self.y_y_mean - 2 * x_y_mean
        return torch.sqrt(maximum_mean_discrepancy_value)


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
