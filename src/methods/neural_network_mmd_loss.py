from pathlib import Path
import torch
from utils.metrics import calculate_rbf_gamma
from utils.models import WeightingMlp
from utils.loss import WeightedMMDLoss
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def neural_network_mmd_loss_weighting(N, R, columns, *args, **attributes):
    bias_variable = attributes["bias_variable"]
    bias_values = None
    if bias_variable is not None:
        bias_values = N[bias_variable]
        bias_values = torch.FloatTensor(bias_values.values).to(device)

    tensor_N = torch.DoubleTensor(N[columns].values)
    tensor_R = torch.DoubleTensor(R[columns].values)

    model, mmd_list, mean_list = compute_model(
        tensor_N,
        tensor_R,
        bias_values=bias_values,
    )

    if bias_values is not None:
        attributes["mean_list"].append(mean_list)

    if attributes["mmd_list"] is not None:
        attributes["mmd_list"].append(mmd_list)

    with torch.no_grad():
        model.eval()
        tensor_N = torch.DoubleTensor(N[columns].values).to(device)
        weights = model(tensor_N).cpu().squeeze().numpy().astype(np.float64)

    return weights / sum(weights)


def compute_model(tensor_N, tensor_R, bias_values=None):
    iterations = 1000
    gamma = calculate_rbf_gamma(np.append(tensor_N, tensor_R, axis=0))
    latent_features = tensor_N.shape[1] // 2
    Path("models").mkdir(exist_ok=True, parents=True)
    model_path = Path(f"models/best_model_mmd_loss.pt")
    mmd_list = []
    learning_rate = 0.1
    best_validation_metric = torch.inf
    means = []

    tensor_N = tensor_N.to(device)
    tensor_R = tensor_R.to(device)

    loss_function_train = WeightedMMDLoss(gamma, tensor_N, tensor_R, device)
    validation_loss_function = WeightedMMDLoss(gamma, tensor_N, tensor_R, device)

    mmd_model = WeightingMlp(tensor_N.shape[1], latent_features).to(device)

    optimizer = torch.optim.Adam(
        mmd_model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate / 10, total_steps=iterations)
    for _ in range(iterations):
        mmd_model.train()
        optimizer.zero_grad()

        train_weights = mmd_model(tensor_N)
        loss = loss_function_train(train_weights)

        loss.backward()
        optimizer.step()
        scheduler.step()

        mmd, validation_weights = validate_model(
            tensor_N,
            validation_loss_function,
            mmd_model,
        )

        mmd_list.append(mmd)

        if mmd < best_validation_metric:
            best_validation_metric = mmd
            torch.save(mmd_model.state_dict(), model_path)

        if bias_values is not None:
            validation_weights = validation_weights.to(device)
            positive_value = torch.sum(bias_values * validation_weights.squeeze())
            means.append(positive_value.cpu())

    mmd_model.load_state_dict(torch.load(model_path))
    mmd_model.eval()

    return mmd_model, mmd_list, means


def validate_model(tensor_N, mmd_loss_function, mmd_model):
    mmd_model.eval()
    with torch.no_grad():
        validation_weights = mmd_model(tensor_N)
        mmd = (
            mmd_loss_function(
                validation_weights,
            )
            .cpu()
            .numpy()
        )

    return mmd, validation_weights
