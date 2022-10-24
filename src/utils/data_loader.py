import pandas as pd
import pathlib

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


def load_barometer():
    data_path = f"{file_path}/../../data/debiasing/spanish_barometer_population.csv"
    barometer = pd.read_csv(data_path, index_col="Unnamed: 0")
    columns = [
        "frequency_of_religious_acts",
        "sex",
        "age",
        "education_level",
        "socioeconomics_status",
        "autonomous_community_of_residence",
        "size_of_municipality_of_residence",
        "nationality",
        "marital_status",
        "degree_of_voting_to_change_things",
    ]
    return barometer, columns, None


def load_dataset(dataset_name):
    if dataset_name == "gbs":
        return load_gbs()
    elif dataset_name == "artificial":
        return load_artificial_data()
    elif dataset_name == "census":
        return load_census_data()
    elif dataset_name == "barometer":
        return load_barometer()


def sample(df, sample_size):
    representative = df.sample(sample_size)
    representative["label"] = 0
    non_representative = df.sample(sample_size, weights=df["pi"])
    non_representative["label"] = 1
    return non_representative, representative


def sample_barometer(df, sample_size, use_age_bias):
    internet_group = df[df["use_of_internet"] == 1]
    representative = df.sample(sample_size)
    representative["label"] = 0
    if use_age_bias:
        non_representative = internet_group.sample(sample_size, weights=df["pi"])
    else:
        non_representative = internet_group.sample(sample_size)
    non_representative["label"] = 1
    return non_representative, representative
