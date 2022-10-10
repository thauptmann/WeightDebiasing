from torch import nn
import torch
import numpy as np


class Mlp(nn.Module):
    def __init__(self, number_of_features):
        half_feature_size = int(number_of_features / 2)
        super(Mlp, self).__init__()
        self.encoding = nn.Sequential(
            nn.Linear(number_of_features, half_feature_size),
            nn.ReLU(),
            nn.BatchNorm1d(half_feature_size),
        )
        self.weighting = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(half_feature_size, 1),
            nn.Softplus(),
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
