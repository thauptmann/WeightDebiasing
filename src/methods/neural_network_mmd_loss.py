from pathlib import Path
import torch
import shap
from scipy.spatial.distance import pdist
from utils.models import Mlp
from .loss import WeightedMMDLoss
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from utils.visualisation import plot_line
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def neural_network_mmd_loss_weighting(N, R, columns, *args, **attributes):
    passes = 10000
    save_path = attributes["save_path"]

    tensor_N = torch.FloatTensor(N[columns].values)
    tensor_R = torch.FloatTensor(R[columns].values)

    mmd_model, mmd_list = compute_model(passes, tensor_N, tensor_R)

    plot_line(mmd_list, save_path, "MMDs_per_pass")

    with torch.no_grad():
        weights = mmd_model(tensor_N).squeeze().numpy()
    # compute_shap_values(mmd_model, tensor_N.numpy(), columns, save_path)
    return weights


def compute_model(
    passes,
    tensor_N,
    tensor_R,
    patience=250,
):
    model_path = Path("best_model.pt")
    mmd_list = []

    gamma = calculate_rbf_gamma(np.append(tensor_N, tensor_R, axis=0))
    mmd_loss_function = WeightedMMDLoss(gamma, len(tensor_R), device)
    learning_rate = 0.001
    early_stopping_counter = 0
    tensor_N = tensor_N.to(device)
    tensor_R = tensor_R.to(device)

    best_mmd = torch.inf
    mmd_model = Mlp(tensor_N.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        mmd_model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(optimizer, patience=patience / 2)
    for _ in trange(passes):
        mmd_model.train()
        optimizer.zero_grad()

        train_weights = mmd_model(tensor_N)
        mmd_loss = mmd_loss_function(tensor_N, tensor_R, train_weights)
        train_weights = train_weights / torch.sum(train_weights)
        if not torch.isnan(mmd_loss) and not torch.isinf(mmd_loss):
            mmd_loss.backward()
            optimizer.step()

        mmd_model.eval()
        with torch.no_grad():
            validation_weights = mmd_model(tensor_N)
        mmd = mmd_loss_function(
            tensor_N,
            tensor_R,
            validation_weights,
        )
        mmd_list.append(mmd.cpu())

        if mmd < best_mmd:
            best_mmd = mmd
            torch.save(mmd_model.state_dict(), model_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter > patience:
                break

        scheduler.step(mmd)

    mmd_model.load_state_dict(torch.load(model_path))
    mmd_model.eval()

    return mmd_model, mmd_list


def compute_shap_values(model, tensor_N, columns, save_path):
    # Compute SHAP values to measure and visualise the bias
    model = model.cpu()
    with torch.no_grad():
        kernelExplainer = shap.Explainer(model, tensor_N, feature_names=columns)
        shap_values = kernelExplainer(tensor_N)
        shap.summary_plot(shap_values, tensor_N, show=False)
        plt.savefig(f"{save_path}/shap_summary.pdf")
        plt.clf()
        shap.plots.bar(shap_values, show=False)
        plt.savefig(f"{save_path}/shap_bars.pdf")
        plt.clf()


def calculate_rbf_gamma(aggregate_set):
    all_distances = pdist(aggregate_set, "euclid")
    sigma = np.median(all_distances)
    return 1 / (2 * (sigma**2))
