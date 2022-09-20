from pathlib import Path
import torch
import shap

from utils.metrics import scale_df
from utils.models import MmdModel
from .loss import AsamLoss, MMDLoss
from .metrics import calculate_rbf_gamma
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from utils.visualisation import plot_line, plot_ratio
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def neural_network_mmd_loss_prediction(df, columns, *args, **attributes):
    model_path = Path("best_model.pt")
    passes = 10000
    save_path = attributes["save_path"]
    bias_variable = attributes.get("bias_variable", None)

    scaled_df, _ = scale_df(columns, df)
    scaled_N = scaled_df[scaled_df["label"] == 1]
    scaled_R = scaled_df[scaled_df["label"] == 0]
    tensor_N = torch.FloatTensor(scaled_N[columns].values)
    tensor_R = torch.FloatTensor(scaled_R[columns].values)

    if bias_variable is not None:
        bias_variable_values = scaled_N[bias_variable]
        representative_variable_values = scaled_R[bias_variable]
        weights_R = np.ones(len(scaled_R)) / len(scaled_R)
        representative_ratio = compute_ratio(
            representative_variable_values.values, weights_R
        )

    else:
        bias_variable_values = None
        representative_variable_values = None

    mmd_model, mmd_list, asam_list, ratio_list = compute_model(
        model_path, passes, tensor_N, tensor_R, bias_variable_values
    )

    plot_line(mmd_list, save_path, "MMDs_per_pass")
    plot_line(asam_list, save_path, "ASAMs_per_pass")
    if bias_variable is not None:
        plot_ratio(ratio_list, representative_ratio, "Ratio_per_pass", save_path)

    with torch.no_grad():
        weights = mmd_model(tensor_N).squeeze().numpy()
    compute_shap_values(mmd_model, tensor_N.numpy(), columns, save_path)

    return weights, None


def compute_model(
    model_path,
    passes,
    tensor_N,
    tensor_R,
    bias_variable=None,
    patience=100,
):
    mmd_list = []
    asam_list = []
    ratio_list = []
    gamma = calculate_rbf_gamma(np.append(tensor_N, tensor_R, axis=0))
    mmd_loss_function = MMDLoss(gamma, len(tensor_R))
    asam_loss_function = AsamLoss()
    learning_rate = 0.001
    early_stopping_counter = 0

    best_mmd = torch.inf
    mmd_model = MmdModel(tensor_N.shape[1]).to(device)
    tensor_N = tensor_N.to(device)
    tensor_R = tensor_R.to(device)
    optimizer = torch.optim.Adam(
        mmd_model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(optimizer, patience=patience/2)
    for _ in trange(passes):
        mmd_model.train()
        optimizer.zero_grad()
        train_weights = mmd_model(tensor_N)
        mmd_loss = mmd_loss_function(tensor_N, tensor_R, train_weights)
        if not torch.isnan(mmd_loss) and not torch.isinf(mmd_loss):
            mmd_loss.backward()
            optimizer.step()

        mmd_model.eval()
        with torch.no_grad():
            validation_weights = mmd_model(tensor_N)
        asam = asam_loss_function(tensor_N, tensor_R, validation_weights)
        asam_list.append(asam)
        mmd = mmd_loss_function(
            tensor_N,
            tensor_R,
            validation_weights,
        )
        mmd_list.append(mmd)

        if mmd < best_mmd:
            best_mmd = mmd
            torch.save(mmd_model.state_dict(), model_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter > patience:
                break

        if bias_variable is not None:
            current_ratio = compute_ratio(
                bias_variable.values, validation_weights.numpy()
            )
            ratio_list.append(current_ratio)
        scheduler.step(mmd)

    mmd_model.load_state_dict(torch.load(model_path))
    mmd_model.eval()

    return mmd_model, mmd_list, asam_list, ratio_list


def compute_shap_values(model, tensor_N, columns, save_path):
    # Compute SHAP values to measure and visualise the bias

    with torch.no_grad():
        kernelExplainer = shap.Explainer(model, tensor_N, feature_names=columns)
        shap_values = kernelExplainer(tensor_N)
        shap.summary_plot(shap_values, tensor_N, show=False)
        plt.savefig(f"{save_path}/shap_summary.pdf")
        plt.clf()
        shap.summary_plot(
            shap_values,
            tensor_N,
            show=False,
            plot_type="violin",
        )
        plt.savefig(f"{save_path}/shap_summary_violin.pdf")
        plt.clf()
        shap.plots.bar(shap_values, show=False)
        plt.savefig(f"{save_path}/shap_bars.pdf")
        plt.clf()


def compute_ratio(bias_values, weights):
    weights = np.squeeze(weights / np.sum(weights))
    one_indices = np.argwhere(bias_values == 1)
    zero_indices = np.argwhere(bias_values == 0)
    positive = np.sum(weights[one_indices])
    negative = np.sum(weights[zero_indices])
    return positive / negative
