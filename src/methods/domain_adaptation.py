import torch
from pathlib import Path
from utils.metrics import calculate_rbf_gamma
from utils.models import Mlp
from utils.loss import WeightedMMDLoss
from torch import nn
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mmd_loss_weight = 0.1


def domain_adaptation_weighting(N, R, columns, *args, **attributes):
    iterations = 1000
    model = compute_model(
        iterations,
        N[columns],
        R[columns],
    )

    with torch.no_grad():
        tensor_N = torch.FloatTensor(N[columns].values).to(device)
        prediction = model(tensor_N).cpu().squeeze()
    predictions = nn.Sigmoid()(prediction)
    weights = (1 - predictions) / predictions
    weights = weights.numpy().astype(np.float64)
    return weights / weights.sum()


def compute_model(
    iterations,
    N,
    R,
):
    best_loss = torch.inf
    tensor_r = torch.FloatTensor(R.values.copy())
    tensor_n = torch.FloatTensor(N.values.copy())
    bce_loss_fn = nn.BCEWithLogitsLoss()

    latent_features = N.shape[1] // 2
    Path("models").mkdir(exist_ok=True, parents=True)
    model_path = Path(f"models/best_model_domain_adaptation.pt")

    gamma = calculate_rbf_gamma(torch.concat([tensor_n, tensor_r]))
    mmd_loss_function = WeightedMMDLoss(gamma, device)

    dataset = torch.concat([tensor_n, tensor_r]).to(device)
    targets = torch.concat([torch.ones(len(tensor_n)), torch.zeros(len(tensor_r))])
    one_idxs = torch.nonzero(targets == 1).squeeze()
    zero_idxs = torch.nonzero(targets == 0).squeeze()

    domain_adaptation_model = Mlp(tensor_n.shape[1], latent_features).to(device)

    learning_rate = 0.1
    optimizer = torch.optim.Adam(
        domain_adaptation_model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate / 10, total_steps=iterations)

    for _ in range(iterations):
        domain_adaptation_model.train()
        optimizer.zero_grad()
        predictions, latent_features = domain_adaptation_model.forward(
            dataset, return_latent=True
        )
        bce_loss = bce_loss_fn(torch.squeeze(predictions).cpu(), targets)
        latent_representative = latent_features[zero_idxs, :]
        latent_non_representative = latent_features[one_idxs, :]

        mmd_loss = mmd_loss_function(latent_representative, latent_non_representative)
        loss = bce_loss + (mmd_loss_weight * mmd_loss)
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            domain_adaptation_model.eval()
            validation_predictions = domain_adaptation_model.forward(dataset)
            validation_loss = accuracy_score(
                validation_predictions.cpu().int(), targets
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                torch.save(domain_adaptation_model.state_dict(), model_path)

    domain_adaptation_model.load_state_dict(torch.load(model_path))
    domain_adaptation_model.eval()

    return domain_adaptation_model
