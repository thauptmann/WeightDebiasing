from pathlib import Path
import torch
from utils.metrics import calculate_rbf_gamma
from utils.models import WeightingMlp
from utils.loss import WeightedMMDLoss
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def neural_network_mmd_loss_weighting(N, R, columns, *args, **attributes):
    iterations = 1000
    bias_variable = attributes["bias_variable"]
    bias_values = None
    if bias_variable is not None:
        bias_values = N[bias_variable]

    if bias_values is not None:
        bias_values = torch.FloatTensor(bias_values.values).to(device)

    model, mmd_list, mean_list = compute_model(
        iterations,
        N,
        R,
        bias_values=bias_values,
        columns=columns,
    )

    if bias_values is not None:
        attributes["mean_list"].append(mean_list)
        attributes["mmd_list"].append(mmd_list)

    with torch.no_grad():
        tensor_N = torch.FloatTensor(N[columns].values)
        tensor_N = tensor_N.to(device)
        weights = model(tensor_N).cpu().squeeze().numpy()
    return weights


def compute_model(
    iterations,
    N,
    R,
    bias_values=None,
    columns=None,
):

    tensor_N = torch.FloatTensor(N[columns].values)
    tensor_R = torch.FloatTensor(R[columns].values)
    gamma = calculate_rbf_gamma(np.append(tensor_N, tensor_R, axis=0))
    latent_features = N.shape[1] // 2
    Path("models").mkdir(exist_ok=True, parents=True)
    model_path = Path(f"models/best_model_mmd_loss.pt")
    mmd_list = []
    learning_rate = 0.1
    best_mmd = torch.inf
    means = []

    tensor_N = tensor_N.to(device)
    tensor_R = tensor_R.to(device)

    mmd_loss_function = WeightedMMDLoss(gamma, tensor_R, device)
    mmd_loss_function_train = WeightedMMDLoss(gamma, tensor_R, device)

    if bias_values is not None:
        validation_weights = (torch.ones(len(tensor_N)) / len(tensor_N)).to(device)
        positive_value = torch.sum(bias_values * validation_weights.squeeze())
        means.append(positive_value.cpu())
        start_mmd = mmd_loss_function(tensor_N, tensor_R, validation_weights)
        mmd_list.append(start_mmd.cpu().numpy())

    mmd_model = WeightingMlp(tensor_N.shape[1], latent_features).to(device)

    optimizer = torch.optim.Adam(
        mmd_model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate / 10, total_steps=iterations)
    for _ in range(iterations):
        mmd_model.train()
        optimizer.zero_grad()

        train_weights = mmd_model(tensor_N)
        mmd_loss = mmd_loss_function_train(tensor_N, tensor_R, train_weights)
        if not torch.isnan(mmd_loss) and not torch.isinf(mmd_loss):
            mmd_loss.backward()
            optimizer.step()
            scheduler.step()

        mmd, validation_weights = validate_model(
            tensor_N, tensor_R, mmd_loss_function, mmd_model
        )

        mmd_list.append(mmd)

        if mmd < best_mmd:
            best_mmd = mmd
            torch.save(mmd_model.state_dict(), model_path)

        if bias_values is not None:
            validation_weights = validation_weights.to(device)
            positive_value = torch.sum(bias_values * validation_weights.squeeze())
            means.append(positive_value.cpu())

    mmd_model.load_state_dict(torch.load(model_path))
    mmd_model.eval()

    return mmd_model, mmd_list, means


def validate_model(tensor_N, tensor_R, mmd_loss_function, mmd_model):
    mmd_model.eval()
    with torch.no_grad():
        validation_weights = mmd_model(tensor_N)
        if torch.sum(validation_weights) == 0:
            mmd = np.nan
        else:
            mmd = mmd_loss_function(
                tensor_N,
                tensor_R,
                validation_weights,
            )
            mmd = mmd.cpu().numpy()

    return mmd, validation_weights
