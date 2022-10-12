from pathlib import Path
import torch
from scipy.spatial.distance import pdist
from utils.models import Mlp
from .loss import MMDLoss
import numpy as np
from tqdm import trange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def domain_adaptation_weighting(N, R, columns, *args, **attributes):
    epochs = 1000
    tensor_N = torch.FloatTensor(N[columns].values)
    tensor_R = torch.FloatTensor(R[columns].values)

    domain_adaptation_model = compute_model(epochs, tensor_N, tensor_R)

    with torch.no_grad():
        tensor_N = tensor_N.to(device)
        predictions = domain_adaptation_model(tensor_N).cpu().squeeze()
    predictions = nn.Sigmoid()(predictions)
    weights = (1 - predictions) / predictions

    return weights


def compute_model(
    epochs,
    tensor_n,
    tensor_r,
    patience=100,
):
    model_path = Path("best_model.pt")

    gamma = calculate_rbf_gamma(torch.concat([tensor_n, tensor_r]))
    mmd_loss_function = MMDLoss(gamma, device)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    learning_rate = 0.001
    early_stopping_counter = 0
    batch_size = 128

    tensor_r = tensor_r.to(device)
    tensor_n = tensor_n.to(device)

    dataset = TensorDataset(
        torch.concat([tensor_n, tensor_r]),
        torch.concat([torch.ones(len(tensor_n)), torch.zeros(len(tensor_r))]),
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    best_validation_loss = torch.inf

    domain_adaptation_model = Mlp(tensor_n.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        domain_adaptation_model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(optimizer, patience=int(patience / 2))
    mmd_loss_weight = 0.1

    for _ in trange(epochs):
        domain_adaptation_model.train()
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            (
                predictions,
                latent_features,
            ) = domain_adaptation_model.forward_with_latent_features(x)
            bce_loss = bce_loss_fn(torch.squeeze(predictions), y)
            one_idxs = torch.nonzero(y == 1).squeeze()
            zero_idxs = torch.nonzero(y == 0).squeeze()
            mmd_loss = mmd_loss_function(
                latent_features[one_idxs, :], latent_features[zero_idxs, :]
            )
            loss = bce_loss + mmd_loss_weight * mmd_loss
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            domain_adaptation_model.eval()
        with torch.no_grad():
            (
                validation_predictions,
                validation_latent_features,
            ) = domain_adaptation_model.forward_with_latent_features(
                torch.concat([tensor_n, tensor_r])
            )
            bce_loss = bce_loss_fn(
                torch.squeeze(validation_predictions),
                torch.concat(
                    [torch.ones(len(tensor_n)), torch.zeros(len(tensor_r))]
                ).to(device),
            )
            mmd_loss = mmd_loss_function(
                validation_latent_features[:batch_size, :],
                validation_latent_features[batch_size:, :],
            )
            validation_loss = bce_loss + mmd_loss_weight * mmd_loss

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(domain_adaptation_model.state_dict(), model_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter > patience:
                break
    domain_adaptation_model.load_state_dict(torch.load(model_path))
    domain_adaptation_model.eval()

    return domain_adaptation_model


def calculate_rbf_gamma(aggregate_set):
    all_distances = pdist(aggregate_set, "euclid")
    sigma = np.median(all_distances)
    return 1 / (2 * (sigma**2))
