import torch
import numpy as np
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR

from utils.metrics import calculate_rbf_gamma
from utils.models import WeightingMlp
from utils.losses import WeightedMMDLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def neural_network_mmd_loss_weighting(N, R, columns, *args, **attributes):
    """_summary_

    :param N: _description_
    :param R: _description_
    :param columns: _description_
    :return: _description_
    """
    N_copy = N.copy().reset_index()
    N_dropped = N_copy.drop_duplicates(subset=columns)
    indices = N_dropped.index
    tensor_N = torch.DoubleTensor(N_dropped[columns].values)
    tensor_R = torch.DoubleTensor(R[columns].values)

    all_weights = np.zeros(len(N))
    model, mmd_list = compute_model(tensor_N, tensor_R)

    if attributes["mmd_list"] is not None:
        attributes["mmd_list"].append(mmd_list)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model.eval()
            weights = (
                model(tensor_N.to(device), tensor_R.to(device))
                .cpu()
                .squeeze()
                .numpy()
                .astype(np.float64)
            )
    all_weights[indices] = weights
    return all_weights / sum(all_weights)


def compute_model(tensor_N, tensor_R):
    """_summary_

    :param tensor_N: _description_
    :param tensor_R: _description_
    :return: _description_
    """
    iterations = 5000
    scaler = torch.cuda.amp.GradScaler()
    gamma = calculate_rbf_gamma(np.append(tensor_N, tensor_R, axis=0))
    latent_features = tensor_N.shape[1] // 2
    Path("models").mkdir(exist_ok=True, parents=True)
    model_path = Path(f"models/best_model_mmd_loss.pt")
    mmd_list = []
    learning_rate = 1
    best_validation_metric = torch.inf

    loss_function = WeightedMMDLoss(gamma, tensor_N, tensor_R, device)
    tensor_N = tensor_N.to(device)
    tensor_R = tensor_R.to(device)

    mmd_model = WeightingMlp(tensor_N.shape[1], latent_features).to(device)
    optimizer = torch.optim.Adam(mmd_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate / 10, total_steps=iterations)

    for _ in range(iterations):
        mmd_model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            train_weights = mmd_model(tensor_N, tensor_R)
            loss = loss_function(train_weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        mmd = validate_model(
            tensor_N,
            tensor_R,
            loss_function,
            mmd_model,
        )
        mmd_list.append(mmd)

        if mmd < best_validation_metric:
            best_validation_metric = mmd
            torch.save(mmd_model.state_dict(), model_path)

    mmd_model.load_state_dict(torch.load(model_path))

    return mmd_model, mmd_list


def validate_model(tensor_N, tensor_R, mmd_loss_function, mmd_model):
    """_summary_

    :param tensor_N: _description_
    :param tensor_R: _description_
    :param mmd_loss_function: _description_
    :param mmd_model: _description_
    :return: _description_
    """
    mmd_model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            validation_weights = mmd_model(tensor_N, tensor_R)
            return (
                mmd_loss_function(
                    validation_weights,
                )
                .cpu()
                .numpy()
            )
