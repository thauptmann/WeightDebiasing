import argparse
from pathlib import Path
import pathlib
import numpy as np
from scipy.stats import bernoulli, norm
import pandas as pd

bernoulli_p = 0.5
random_seed = 5
file_path = pathlib.Path(__file__).parent
np.random.seed(random_seed)

columns = [
    "x_1",
    "x_2",
    "x_3",
    "x_4",
    "x_5",
    "x_6",
    "x_7",
    "x_8",
    "y_1",
    "y_2",
    "y_3",
    "y_4",
    "y_5",
    "y_6",
    "y_7",
    "y_8",
    "pi",
]


def create_aritficial_data_set(size, filename):
    save_path = Path("../../data/debiasing/")
    save_path.mkdir(exist_ok=True, parents=True)

    x_1 = bernoulli.rvs(bernoulli_p, size=size)
    x_2 = create_collinearity_normal_distribution(x_1)
    x_3 = bernoulli.rvs(bernoulli_p - 0.2, size=size)
    x_4 = norm.rvs(loc=2, scale=1, size=size)
    x_4 = create_correlated_normal_distribution(x_3, x_4, 0.2)
    x_5 = bernoulli.rvs(bernoulli_p + 0.2, size=size)
    x_6 = norm.rvs(loc=2, scale=2, size=size)
    x_7 = bernoulli.rvs(bernoulli_p, size=size)
    x_8 = norm.rvs(loc=1, scale=1, size=size)
    x_8 = create_correlated_normal_distribution(x_7, x_8, 0.5)

    logit_i = -0.5 + x_5 + (2.5 * x_6 * x_8) - x_7

    y_1 = bernoulli.rvs(bernoulli_p, size=size)
    y_2 = norm.rvs(loc=10, scale=1, size=size)

    y_3_p = np.exp(logit_i) / (1 + np.exp(logit_i))
    y_3 = np.squeeze([bernoulli.rvs(p, size=1) for p in y_3_p])

    y_4 = norm.rvs(loc=10, scale=1, size=size) + (5 * logit_i)

    invert_x_5 = np.zeros_like(x_5)
    invert_x_5[x_5 == 0] = 1
    logits_5 = 0.5 + (0.25 * x_5) - (0.25 * invert_x_5) + x_6

    y_5_p = np.exp(logits_5) / (1 + np.exp(logits_5))
    y_5 = np.squeeze([bernoulli.rvs(p, size=1) for p in y_5_p])
    y_6 = norm.rvs(loc=10, scale=1, size=size) + (2 * x_5) - (2 * invert_x_5) + x_6

    invert_x_7 = np.zeros_like(x_7)
    invert_x_7[x_7 == 0] = 1
    logit_7 = 0.5 + (0.25 * x_7) - (0.25 * invert_x_7) + x_8 + logit_i
    y_7_p = np.exp(logit_7) / (1 + np.exp(logit_7))
    y_7 = np.squeeze([bernoulli.rvs(p, size=1) for p in y_7_p])

    y_8 = (
        norm.rvs(loc=10, scale=1, size=size)
        + (2 * x_7)
        - (2 * invert_x_7)
        + x_8
        + (5 * logit_i)
    )

    p_i = np.exp(logit_i) / (1 + np.exp(logit_i))
    samples = np.stack(
        [
            x_1,
            x_2,
            x_3,
            x_4,
            x_5,
            x_6,
            x_7,
            x_8,
            y_1,
            y_2,
            y_3,
            y_4,
            y_5,
            y_6,
            y_7,
            y_8,
            p_i,
        ],
        axis=1,
    )

    dataframe = pd.DataFrame(samples, columns=columns)
    dataframe.to_csv(f"{save_path}/{filename}.csv")


def create_collinearity_normal_distribution(x, loc=2, scale=1):
    normal_samples = np.zeros_like(x, dtype=np.float64)
    zero_indices = (x == 0).nonzero()[0]
    one_indices = (x == 1).nonzero()[0]
    normal_samples[zero_indices] = norm.rvs(loc=0, scale=1, size=len(zero_indices))
    normal_samples[one_indices] = norm.rvs(loc=loc, scale=scale, size=len(one_indices))
    return normal_samples


def create_correlated_normal_distribution(x, y, p):
    correlated_samples = (p * x) + (np.sqrt(1 - (p**2)) * y)
    return correlated_samples


barometer_cols = ["P44", "P43", "P1", "P4", "P0", "P0A", "P58"]
map_cols_to_understandable_names = {
    "P44": "age",
    "P43": "sex",
    "P1": "economical_situation_spain",
    "P4": "economical_situation_personal",
    "P0": "nationality",
    "P58": "use_of_internet"
}

replace_values = {
    "Hombre": 0,
    "Mujer": 1,
    "Mala": 1,
    "Muy mala": 1,
    "Regular": 0,
    "Buena": 0,
    "No": 0,
    "Sí": 1
}


def create_barometer_population(size, filename):
    save_path = Path(f"{file_path}/../../data/debiasing/")
    save_path.mkdir(exist_ok=True, parents=True)

    spss = pd.read_spss(f"{save_path}/spanish_barometer.sav", usecols=barometer_cols)
    spss = spss.rename(columns=map_cols_to_understandable_names)
    spss = spss.replace(replace_values)

    # spss_reduced_columns =
    spss = spss.sample(size, replace=True)
    spss.to_csv(f"{save_path}/{filename}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--size", default=50000, type=int, required=True)
    parser.add_argument("--filename", default=None, type=str, required=True)
    args = parser.parse_args()
    data_set = args.dataset
    if data_set == "artificial":
        create_aritficial_data_set(args.size, args.filename)
    else:
        create_barometer_population(args.size, args.filename)
