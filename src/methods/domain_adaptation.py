from pathlib import Path
from sklearn.model_selection import KFold
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


def domain_adaptation_weighting(N, R, columns, number_of_splits, *args, **attributes):
    predictions = np.zeros(len(N))
    k_fold = KFold(n_splits=number_of_splits, shuffle=True)
    epochs = 1000
    for train_index, test_index in k_fold.split(N):
        train_N = N.iloc[train_index]
        test_N = N.iloc[test_index]
        tensor_N = torch.FloatTensor(train_N[columns].values)
        number_of_features = tensor_N.shape[1]
        tensor_test_N = torch.FloatTensor(test_N[columns].values)
        
        tensor_R = torch.FloatTensor(R[columns].values)

        latent_feature_list = [
            number_of_features,
            int(number_of_features * 0.75),
            int(number_of_features * 1.25),
        ]
        best_loss = np.inf
        best_model = None

        for latent_features in latent_feature_list:
            domain_adaptation_model, loss = compute_model(
                epochs, tensor_N, tensor_R, 100, latent_features
            )

            if loss < best_loss:
                best_model = domain_adaptation_model

        with torch.no_grad():
            tensor_test_N = tensor_test_N.to(device)
            prediction = best_model(tensor_test_N).cpu().squeeze()
        predictions[test_index] = nn.Sigmoid()(prediction)
    weights = (1 - predictions) / predictions

    return weights


def compute_model(
    epochs,
    tensor_n,
    tensor_r,
    patience=100,
    latent_features=1
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

    domain_adaptation_model = Mlp(tensor_n.shape[1], latent_features).to(device)
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

    return domain_adaptation_model, best_validation_loss


def calculate_rbf_gamma(aggregate_set):
    all_distances = pdist(aggregate_set, "euclid")
    sigma = np.median(all_distances)
    return 1 / (2 * (sigma**2))
