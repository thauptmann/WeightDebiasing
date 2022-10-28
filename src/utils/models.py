from torch import nn
import torch
import numpy as np


class WeightingMlp(nn.Module):
    def __init__(self, number_of_features, latent_features, dropout):
        super(WeightingMlp, self).__init__()
        self.encoding = nn.Sequential(
            nn.Linear(number_of_features, latent_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(latent_features),
        )
        self.weighting = nn.Sequential(
            nn.Linear(latent_features, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        if isinstance(x, (np.ndarray)):
            x = torch.FloatTensor(x)
        latent_features = self.encoding(x)
        return self.weighting(latent_features)

    def forward_with_latent_features(self, x):
        if isinstance(x, (np.ndarray)):
            x = torch.FloatTensor(x)
        latent_features = self.encoding(x)
        return self.weighting(latent_features), latent_features


class Mlp(nn.Module):
    def __init__(self, number_of_features, latent_features):
        super(Mlp, self).__init__()
        self.encoding = nn.Sequential(
            nn.Linear(number_of_features, latent_features),
            nn.ReLU(),
            nn.BatchNorm1d(latent_features),
        )
        self.weighting = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(latent_features, 1),
        )

    def forward(self, x):
        if isinstance(x, (np.ndarray)):
            x = torch.FloatTensor(x)
        latent_features = self.encoding(x)
        return self.weighting(latent_features)

    def forward_with_latent_features(self, x):
        if isinstance(x, (np.ndarray)):
            x = torch.FloatTensor(x)
        latent_features = self.encoding(x)
        return self.weighting(latent_features), latent_features
