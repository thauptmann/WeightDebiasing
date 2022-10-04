from torch import nn
import torch
import numpy as np


class MmdModel(nn.Module):
    def __init__(self, number_of_features):
        half_feature_size = int(number_of_features / 2)
        super(MmdModel, self).__init__()
        output_linear = nn.Linear(half_feature_size, 1)
        output_linear.bias = nn.Parameter(torch.FloatTensor([1 / 500]))
        self.weight_stack = nn.Sequential(
            nn.Linear(number_of_features, half_feature_size),
            nn.ReLU(),
            nn.BatchNorm1d(half_feature_size),
            nn.Dropout(0.1),
            output_linear,
            nn.Softplus(),
        )

    def forward(self, x):
        if isinstance(x, (np.ndarray)):
            x = torch.FloatTensor(x)
        return self.weight_stack(x)
