from pathlib import Path
import torch
import shap
from torch.utils.data import DataLoader

from utils.metrics import scale_df
from utils.models import MmdModel
from .loss import AsamLoss, MMDLoss
from .metrics import (
    maximum_mean_discrepancy_weighted,
    calculate_rbf_gamma,
)
import numpy as np
from tqdm import trange, tqdm
from utils.data_set import CsvDataset
import matplotlib.pyplot as plt


def neural_network_mmd_loss_prediction(
    df, number_of_splits, columns, not_used, *args, **attributes
):
    model_path = Path("best_model.pt")
    epochs = 10
    data_set_name = attributes["data_set_name"]
    scaled_df, _ = scale_df(columns, df)
    scaled_N = scaled_df[scaled_df["label"] == 1]
    scaled_R = scaled_df[scaled_df["label"] == 0]
    tensor_N = torch.FloatTensor(scaled_N[columns].values)
    tensor_R = torch.FloatTensor(scaled_R[columns].values)
    batch_size = len(tensor_N)

    training_data = CsvDataset(tensor_N)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    mmd_model = compute_model(model_path, epochs, tensor_N, tensor_R, train_dataloader)

    # Compute SHAP values to measure the bias
    f = lambda x: mmd_model(torch.FloatTensor(torch.from_numpy(x))).detach().numpy()

    with torch.no_grad():
        weights = mmd_model(tensor_N).squeeze().numpy()
    #    kernelExplainer = shap.KernelExplainer(f, tensor_N[: 50].numpy())
    #shap_values = kernelExplainer.shap_values(tensor_N.numpy())
    #shap.summary_plot(shap_values[0], tensor_N, columns, show=False)
    plt.savefig(f"../results/neural_network_mmd_loss/{data_set_name}/shap_summary.pdf")

    return weights, None


def compute_model(
    model_path, epochs, tensor_N, tensor_R, train_dataloader, number_of_samples=100
):
    gamma = calculate_rbf_gamma(np.append(tensor_N, tensor_R, axis=0))
    mmd_loss_function = MMDLoss(gamma)
    asam_loss_function = AsamLoss()
    learning_rate = 0.001
    mmd_metric_function = maximum_mean_discrepancy_weighted

    best_mmd = torch.inf
    mmd_model = MmdModel(tensor_N.shape[1])
    optimizer = torch.optim.Adam(
        mmd_model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    for i in trange(epochs):
        mmd_model.train()
        for _ in range(number_of_samples):
            indices = np.random.choice(
                tensor_N.shape[0], size=tensor_N.shape[0], replace=True
            )
            x = tensor_N[indices]
            indices = np.random.choice(
                tensor_N.shape[0], size=tensor_N.shape[0], replace=True
            )
            y = tensor_R[indices]

            optimizer.zero_grad()
            weights = mmd_model(tensor_N)
            mmd_loss = mmd_loss_function(tensor_N, tensor_R, weights)
            # asam_loss = asam_loss_function(x, y, weights)
            # combined_loss = mmd_loss + asam_loss
            if not torch.isnan(mmd_loss) and not torch.isinf(mmd_loss):
                mmd_loss.backward()
                optimizer.step()

        mmd_model.eval()
        with torch.no_grad():
            weights = mmd_model(tensor_N)
        asam = asam_loss_function(tensor_N, tensor_R, weights)
        tqdm.write(f"asam: {asam.numpy()}")

        mmd = mmd_metric_function(tensor_N, tensor_R, weights.squeeze().numpy(), gamma)
        tqdm.write(f"{mmd=}")
        if mmd < best_mmd:
            best_mmd = mmd
            torch.save(mmd_model.state_dict(), model_path)

    mmd_model.load_state_dict(torch.load(model_path))
    mmd_model.eval()

    return mmd_model
