from sklearn.model_selection import KFold
import torch
from scipy.spatial.distance import pdist
from utils.models import Mlp
from utils.loss import MMDLoss
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import ray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout_list = [0.0, 0.2, 0.4]
learning_rate = 0.001
batch_size = 32
mmd_loss_weight = 0.1


def domain_adaptation_weighting(N, R, columns, number_of_splits, *args, **attributes):
    epochs = 1000
    number_of_features = N.shape[1]
    latent_feature_list = [
        number_of_features,
        number_of_features // 2,
        number_of_features // 4,
    ]
    patience = 100

    futures = [
        compute_model.remote(
            epochs,
            N[columns],
            R[columns],
            patience,
            latent_features,
            dropout,
            number_of_splits,
        )
        for latent_features in latent_feature_list
        for dropout in dropout_list
    ]

    results = ray.get(futures)
    loss = [result[1] for result in results]
    max_value = max(loss)
    max_index = loss.index(max_value)

    best_latent_features = results[max_index][2]
    best_dropout = results[max_index][3]

    # Retrain best model
    best_model = train_best_model(
        epochs, N, R, patience, best_latent_features, best_dropout
    )

    with torch.no_grad():
        tensor_N = tensor_N.to(device)
        prediction = best_model(tensor_N).cpu().squeeze()
    predictions = nn.Sigmoid()(prediction)
    weights = (1 - predictions) / predictions

    return weights


@ray.remote(num_gpus=0.5)
def compute_model(epochs, N, R, patience, latent_features, dropout, number_of_splits):
    k_fold = KFold(n_splits=number_of_splits, shuffle=True)
    result_list = []
    tensor_r = torch.FloatTensor(R.values).to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    for train_index, test_index in k_fold.split(N):
        train_N = N.iloc[train_index]
        test_N = N.iloc[test_index]
        tensor_N = torch.FloatTensor(train_N.values)
        tensor_test_N = torch.FloatTensor(test_N.values)
        gamma = calculate_rbf_gamma(torch.concat([train_N, tensor_r]))
        mmd_loss_function = MMDLoss(gamma, device)

        tensor_n = tensor_N.to(device)

        dataset = TensorDataset(
            torch.concat([tensor_n, tensor_r]),
            torch.concat([torch.ones(len(tensor_n)), torch.zeros(len(tensor_r))]),
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        domain_adaptation_model = Mlp(tensor_n.shape[1], latent_features, dropout).to(
            device
        )
        optimizer = torch.optim.Adam(
            domain_adaptation_model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = ReduceLROnPlateau(optimizer, patience=(patience // 2))

        for _ in range(epochs):
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
                loss = bce_loss - (mmd_loss_weight * mmd_loss)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                domain_adaptation_model.eval()

        domain_adaptation_model.eval()
        with torch.no_grad():
            (
                validation_predictions,
                _,
            ) = domain_adaptation_model(torch.concat([tensor_test_N, tensor_r]))
            bce_loss = bce_loss_fn(
                torch.squeeze(validation_predictions),
                torch.concat(
                    [torch.ones(len(tensor_test_N)), torch.zeros(len(tensor_r))]
                ).to(device),
            )
            validation_loss = bce_loss
            result_list.append(validation_loss)

    return domain_adaptation_model, torch.mean(result_list), latent_features, dropout


def calculate_rbf_gamma(aggregate_set):
    all_distances = pdist(aggregate_set, "euclid")
    sigma = np.median(all_distances)
    return 1 / (2 * (sigma**2))


def train_best_model(epochs, N, R, patience, latent_features, dropout):
    tensor_n = torch.FloatTensor(N.values).to(device)
    tensor_r = torch.FloatTensor(R.values).to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    gamma = calculate_rbf_gamma(torch.concat([tensor_n, tensor_r]))
    mmd_loss_function = MMDLoss(gamma, device)

    dataset = TensorDataset(
        torch.concat([tensor_n, tensor_r]),
        torch.concat([torch.ones(len(tensor_n)), torch.zeros(len(tensor_r))]),
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    domain_adaptation_model = Mlp(tensor_n.shape[1], latent_features, dropout).to(
        device
    )
    optimizer = torch.optim.Adam(
        domain_adaptation_model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(optimizer, patience=(patience // 2))

    for _ in range(epochs):
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
            loss = bce_loss + (mmd_loss_weight * mmd_loss)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

    domain_adaptation_model.eval()

    return domain_adaptation_model
