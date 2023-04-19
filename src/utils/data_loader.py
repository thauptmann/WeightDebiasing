import pandas as pd
import pathlib
from folktables import ACSDataSource, generate_categories, BasicProblem
import folktables
import numpy as np

file_path = pathlib.Path(__file__).parent


def load_gbs():
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
    return allensbach, allensbach_columns


def load_artificial_data():
    artificial_data_path = f"{file_path}/../../data/debiasing/artificial_population.csv"
    artificial = pd.read_csv(artificial_data_path, index_col="Unnamed: 0")
    columns = artificial.filter(like="x").columns
    return artificial, columns


def load_census_data(census_bias="Above_Below 50K"):
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
        na_values=["-1", "-1", " ?"],
    )
    df, preprocessed_columns = preprocess_census(df, census_bias)
    return df, preprocessed_columns


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
        "other",
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
    df = df[df["Workclass"] != " Without-pay"]
    df = df.replace(" Self-emp-inc", "Self-emp")
    df = df.replace(" Self-emp-not-inc", "Self-emp")
    df = df.replace(" Asian-Pac-Islander", " Other")
    df = df.replace(" Local-gov", "Gov")
    df = df.replace(" State-gov", "Gov")

    ctg = ["Workclass", "Marital Status", "Race", "Country"]
    for c in ctg:
        df = pd.concat(
            [df, pd.get_dummies(df[c], prefix=c, dummy_na=False, drop_first=True)],
            axis=1,
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

    return df, census_columns


def load_dataset(dataset_name, census_bias):
    if dataset_name == "gbs":
        return load_gbs()
    elif dataset_name == "artificial":
        return load_artificial_data()
    elif dataset_name == "census":
        return load_census_data(census_bias)
    elif dataset_name == "folktables":
        return load_folktables_data()
    elif dataset_name == "mrs_census":
        return load_mrs_census_data()


def sample(df, bias_sample_size, reference_sample_size=1000):
    representative = df.sample(reference_sample_size)
    representative["label"] = 0
    non_representative = df.sample(bias_sample_size, weights=df["pi"])
    non_representative["label"] = 1
    return non_representative, representative


def sample_folk(country, state, bias_sample_size, reference_sample_size=2000):
    representative = country.sample(reference_sample_size)
    non_representative = state.sample(bias_sample_size)
    return non_representative, representative


def load_folktables_data():
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    ca_data = data_source.get_data(states=["CA"], download=True)
    us_data = data_source.get_data(states=["CA"], download=True)
    definition_df = data_source.get_definitions(download=True)
    categories = generate_categories(
        features=ACSIncomeNew.features, definition_df=definition_df
    )
    state_features, state_labels, _ = ACSIncomeNew.df_to_pandas(
        ca_data, categories=categories, dummies=True
    )

    us_features, us_labels, _ = ACSIncomeNew.df_to_pandas(
        us_data, categories=categories, dummies=True
    )

    columns = state_features.columns
    us_features["label"] = 0
    state_features["label"] = 1

    us_features["PINCP"] = us_labels
    state_features["PINCP"] = state_labels

    df = pd.concat([us_features, state_features])
    df = df.dropna()

    return df, columns


ACSIncomeNew = BasicProblem(
    features=[
        "AGEP",
        "COW",
        "SCHL",
        "MAR",
        "OCCP",
        "POBP",
        "RELP",
        "WKHP",
        "SEX",
        "RAC1P",
    ],
    target="PINCP",
    preprocess=folktables.adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)


def load_mrs_census_data():
    census_bias = "Marital Status_Married"
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
        "Hours/Week",
        "Country",
        "Above_Below 50K",
    ]

    df = pd.read_csv(
        f"{file_path}/../../data/Census_Income/adult.data",
        names=columns,
        na_values=["-1", "-1", " ?"],
    )

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

    ctg = ["Workclass", "Marital Status", "Occupation", "Race", "Country"]

    for c in ctg:
        df = pd.concat(
            [df, pd.get_dummies(df[c], prefix=c, dummy_na=False)], axis=1
        ).drop([c], axis=1)

    census_columns = list(df.columns)
    meta = ["label", "index", "fnlgwt", "Education", "Relationship", census_bias]
    for m in meta:
        if m in census_columns:
            census_columns.remove(m)

    df = df.drop(["Education", "Relationship"], axis="columns")
    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)

    return df, census_columns
