from torch import nn
import torch
import numpy as np


class WeightingMlp(nn.Module):
    def __init__(self, number_of_features, latent_features):
        super(WeightingMlp, self).__init__()
        self.weighting = nn.Sequential(
            nn.Linear(number_of_features, latent_features, dtype=torch.float64),
            nn.BatchNorm1d(latent_features, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(latent_features, 1, dtype=torch.float64),
        )
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x, z):
        weights = self.weighting(x).flatten()
        return self.softmax(weights)


class Mlp(nn.Module):
    def __init__(self, number_of_features, latent_features):
        super(Mlp, self).__init__()
        self.encoding = nn.Sequential(
            nn.Linear(number_of_features, latent_features),
            nn.ReLU(),
            nn.BatchNorm1d(latent_features),
        )
        self.weighting = nn.Sequential(
            nn.Linear(latent_features, 1),
        )

    def forward(self, x, return_latent=False):
        if isinstance(x, (np.ndarray)):
            x = torch.FloatTensor(x)
        latent_features = self.encoding(x)
        if return_latent:
            return self.weighting(latent_features), latent_features
        else:
            return self.weighting(latent_features)
