from pathlib import Path
import torch
from utils.metrics import calculate_rbf_gamma, WassersteinMetric
from utils.models import WeightingMlp
from utils.loss import WeightedMMDLoss, WassersteinLoss, WeightedMMDWassersteinLoss
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def neural_network_mmd_loss_weighting(
    N, R, columns, loss_function, *args, **attributes
):
    iterations = 1000
    bias_variable = attributes["bias_variable"]
    bias_values = None
    if bias_variable is not None:
        bias_values = N[bias_variable]
        bias_values = torch.FloatTensor(bias_values.values).to(device)

    model, mmd_list, mean_list, wasserstein_list = compute_model(
        iterations,
        N,
        R,
        bias_values=bias_values,
        columns=columns,
        loss_function=loss_function,
    )

    if bias_values is not None:
        attributes["mean_list"].append(mean_list)

    if attributes["mmd_list"] is not None:
        attributes["mmd_list"].append(mmd_list)
        attributes["wasserstein_list"].append(wasserstein_list)

    with torch.no_grad():
        tensor_N = torch.DoubleTensor(N[columns].values)
        tensor_N = tensor_N.to(device)
        weights = model(tensor_N).cpu().squeeze().numpy().astype(np.float64)
    return weights


def compute_model(iterations, N, R, bias_values=None, columns=None, loss_function=None):

    tensor_N = torch.DoubleTensor(N[columns].values)
    tensor_R = torch.DoubleTensor(R[columns].values)
    gamma = calculate_rbf_gamma(np.append(tensor_N, tensor_R, axis=0))
    latent_features = N.shape[1] // 2
    Path("models").mkdir(exist_ok=True, parents=True)
    model_path = Path(f"models/best_model_mmd_loss.pt")
    mmd_list = []
    wasserstein_list = []
    learning_rate = 0.1
    best_validation_metric = torch.inf
    means = []
    compute_emd = False

    tensor_N = tensor_N.to(device)
    tensor_R = tensor_R.to(device)

    if loss_function == "mmd_rbf":
        loss_function_train = WeightedMMDLoss(gamma, tensor_R, device)
        mmd_loss_function = WeightedMMDLoss(gamma, tensor_R, device)
    elif loss_function == "mmd_linear":
        loss_function_train = WeightedMMDLoss(gamma, tensor_R, device, "linear")
        mmd_loss_function = WeightedMMDLoss(gamma, tensor_R, device, "linear")
    else:
        loss_function_train = WassersteinLoss(tensor_R, device)
        compute_emd = True

    mmd_model = WeightingMlp(tensor_N.shape[1], latent_features).to(device)

    optimizer = torch.optim.Adam(
        mmd_model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate / 10, total_steps=iterations)
    for _ in range(iterations):
        mmd_model.train()
        optimizer.zero_grad()

        train_weights = mmd_model(tensor_N)
        loss = loss_function_train(
            tensor_N,
            tensor_R,
            train_weights,
        )

        if not torch.isnan(loss) and not torch.isinf(loss) and loss != 0:
            loss.backward()
            optimizer.step()
            scheduler.step()

        mmd, wasserstein, validation_weights = validate_model(
            tensor_N,
            tensor_R,
            mmd_loss_function,
            mmd_model,
            compute_emd,
        )

        mmd_list.append(mmd)
        wasserstein_list.append(wasserstein)

        if loss_function == "mmd":
            validation_metric = mmd
        elif loss_function == "wasserstein":
            validation_metric = wasserstein
        else:
            validation_metric = wasserstein + mmd

        if validation_metric < best_validation_metric:
            best_validation_metric = validation_metric
            torch.save(mmd_model.state_dict(), model_path)

        if bias_values is not None:
            validation_weights = validation_weights.to(device)
            positive_value = torch.sum(bias_values * validation_weights.squeeze())
            means.append(positive_value.cpu())

    mmd_model.load_state_dict(torch.load(model_path))
    mmd_model.eval()

    return mmd_model, mmd_list, means, wasserstein_list


def validate_model(tensor_N, tensor_R, mmd_loss_function, mmd_model, compute_emd=False):
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

        if compute_emd:
            wasserstein = (
                WassersteinMetric(
                    tensor_N,
                    tensor_R,
                    validation_weights,
                    method="sinkhorn",
                )
                .cpu()
                .numpy()
            )
        else:
            wasserstein = 0

    return mmd, wasserstein, validation_weights
