import pandas as pd
import pathlib

dataset_list = ["allensbach", "gesis", "artificial", "census"]
file_path = pathlib.Path(__file__).parent.resolve()


def load_allensbach():
    allensbach_path = f"{file_path}/../../data/allensbach_mrs.csv"
    allensbach = pd.read_csv(allensbach_path)
    allensbach.drop(["Unnamed: 0", "Gruppe", "GBS-CODE"], axis=1, inplace=True)
    allensbach_columns = [
        "Alter",
        "Berufsgruppe",
        "Erwerbstaetigkeit",
        "Geschlecht",
        "Optimismus",
        "Pessimismus",
        "Schulabschluss",
        "woechentlicheArbeitszeit",
        "Resilienz",
    ]
    return allensbach, allensbach_columns, None


def load_artificial_data():
    artificial_data_path = f"{file_path}/../../data/debiasing/artificial.csv"
    artificial = pd.read_csv(artificial_data_path, index_col="Unnamed: 0")
    columns = artificial.filter(like="x").columns
    return artificial, columns, None


def load_census_data():
    census_bias = "Above_Below 50K"
    columns = [
        "Age",
        "Workclass",
        "fnlgwt",
        "Education",
        "Education Num",
        "Marital Status",
        "Occupation",
        "Relationship",
        "Race",
        "Sex",
        "Capital Gain",
        "Capital Loss",
        "Hours_per_Week",
        "Country",
        "Above_Below 50K",
    ]
    df = pd.read_csv(
        f"{file_path}/../../data/Census_Income/adult.data",
        names=columns,
        na_values=["-1", -1, " ?"],
    )
    df, preprocessed_columns = preprocess_census(df, census_bias)
    return df, preprocessed_columns, census_bias


def preprocess_census(df, census_bias):
    df = df.replace(
        [
            " Cambodia",
            " China",
            " Hong",
            " Laos",
            " Thailand",
            " Japan",
            " Taiwan",
            " Vietnam",
            " Philippines",
            " India",
            " Iran",
            " Cuba",
            " Guatemala",
            " Jamaica",
            " Nicaragua",
            " Puerto-Rico",
            " Dominican-Republic",
            " El-Salvador",
            " Haiti",
            " Honduras",
            " Mexico",
            " Trinadad&Tobago",
            " Ecuador",
            " Peru",
            " Columbia",
            " South",
            " Poland",
            " Yugoslavia",
            " Hungary",
            " Outlying-US(Guam-USVI-etc)",
        ],
        "other",
    )
    df = df.replace(
        [
            " England",
            " Germany",
            " Holand-Netherlands",
            " Ireland",
            " France",
            " Greece",
            " Italy",
            " Portugal",
            " Scotland",
        ],
        "west_europe",
    )
    df = df.replace(
        [" Married-civ-spouse", " Married-spouse-absent", " Married-AF-spouse"],
        "Married",
    )
    df.replace(" >50K.", 1, inplace=True)
    df.replace(" >50K", 1, inplace=True)
    df.replace(" <=50K.", 0, inplace=True)
    df.replace(" <=50K", 0, inplace=True)
    df["Sex"].replace(" Male", 1, inplace=True)
    df["Sex"].replace(" Female", 0, inplace=True)
    df.dropna(inplace=True)
    ctg = ["Workclass", "Marital Status", "Race", "Country"]
    for c in ctg:
        df = pd.concat(
            [df, pd.get_dummies(df[c], prefix=c, dummy_na=False)], axis=1
        ).drop([c], axis=1)
    census_columns = list(df.columns)
    meta = [
        "label",
        "index",
        "fnlgwt",
        "Education",
        "Relationship",
        "Occupation",
        census_bias,
    ]
    for m in meta:
        if m in census_columns:
            census_columns.remove(m)
    df = df.drop(["fnlgwt", "Education", "Relationship", "Occupation"], axis="columns")

    equal_probability = 1 / len(df)
    df["weights"] = equal_probability - (df[census_bias] * (equal_probability * 0.5))
    fraction = 0.1

    rep = df.sample(frac=fraction)
    nonrep_negative_class = df.sample(frac=fraction, weights=df["weights"])
    rep["label"] = 0
    nonrep_negative_class["label"] = 1
    census_nonrep_more_negative_class = pd.concat(
        [rep.copy(deep=True), nonrep_negative_class.copy(deep=True)]
    )
    return census_nonrep_more_negative_class, census_columns


def load_dataset(dataset_name):
    if dataset_name == "allensbach":
        return load_allensbach()
    elif dataset_name == "artificial":
        return load_artificial_data()
    elif dataset_name == "census":
        return load_census_data()
