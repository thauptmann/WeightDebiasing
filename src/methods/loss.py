import torch
import torch.nn as nn


euclidean_distance_fn = torch.nn.PairwiseDistance(2)


# Weighted MMD loss
class WeightedMMDLoss(nn.Module):
    def __init__(self, gamma, len_R, device):
        super(WeightedMMDLoss, self).__init__()
        self.gamma = gamma
        self.weights_R = (torch.ones(len_R) / len_R).to(device)
        self.device = device

    def rbf_kernel(self, source, target, gamma):
        distance_matrix = torch.cdist(source, target, p=2)
        squared_distance_matrix = distance_matrix.pow(2)
        return torch.exp(-gamma * squared_distance_matrix)

    def calculate_rbf_gamma(self, aggregate_set):
        all_distances = euclidean_distance_fn(aggregate_set, p=2)
        sigma = torch.median(all_distances)
        return 1 / (2 * (sigma**2))

    def forward(self, N, R, weights):
        weights = torch.squeeze(weights)

        if self.gamma is None:
            concated = torch.concat([N, R], dim=0)
            gamma = self.calculate_rbf_gamma(concated)
        else:
            gamma = self.gamma

        x_x_rbf_matrix = torch.matmul(
            torch.unsqueeze(weights, 1), torch.unsqueeze(weights, 0)
        ) * self.rbf_kernel(N, N, gamma)
        x_x_mean = x_x_rbf_matrix.sum()

        y_y_rbf_matrix = torch.matmul(
            torch.unsqueeze(self.weights_R, 1), torch.unsqueeze(self.weights_R, 0)
        ) * self.rbf_kernel(R, R, gamma)
        y_y_mean = y_y_rbf_matrix.sum()
        weight_matrix = torch.matmul(
            torch.unsqueeze(weights, 1), torch.unsqueeze(self.weights_R, 0)
        )
        x_y_rbf_matrix = weight_matrix * self.rbf_kernel(N, R, gamma)
        x_y_mean = x_y_rbf_matrix.sum()

        maximum_mean_discrepancy_value = x_x_mean + y_y_mean - 2 * x_y_mean
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


def earth_mover_distance(p, q):
    x = p - q
    y = torch.cumsum(x, dim=0)
    return abs(y).sum().div(len(p))


# MMD loss
class MMDLoss(nn.Module):
    def __init__(self, gamma, device):
        super(MMDLoss, self).__init__()
        self.gamma = gamma
        self.device = device

    def rbf_kernel(self, source, target, gamma):
        distance_matrix = torch.cdist(source, target, p=2)
        squared_distance_matrix = distance_matrix.pow(2)
        return torch.exp(-gamma * squared_distance_matrix)

    def calculate_rbf_gamma(self, aggregate_set):
        all_distances = euclidean_distance_fn(aggregate_set, p=2)
        sigma = torch.median(all_distances)
        return 1 / (2 * (sigma**2))

    def forward(self, N, R):

        if self.gamma is None:
            concated = torch.concat([N, R], dim=0)
            gamma = self.calculate_rbf_gamma(concated)
        else:
            gamma = self.gamma

        x_x_rbf_matrix = self.rbf_kernel(N, N, gamma)
        x_x_mean = x_x_rbf_matrix.mean()

        y_y_rbf_matrix = self.rbf_kernel(R, R, gamma)
        y_y_mean = y_y_rbf_matrix.mean()
        x_y_rbf_matrix = self.rbf_kernel(N, R, gamma)
        x_y_mean = x_y_rbf_matrix.mean()

        maximum_mean_discrepancy_value = x_x_mean + y_y_mean - 2 * x_y_mean
        return torch.sqrt(maximum_mean_discrepancy_value)
