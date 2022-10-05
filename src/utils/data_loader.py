import pandas as pd

dataset_list = ["allensbach", "gesis", "artificial", "census"]


def load_allensbach():
    allensbach_path = "../data/allensbach_mrs.csv"
    allensbach = pd.read_csv(allensbach_path)
    allensbach.drop(["Unnamed: 0"], axis=1, inplace=True)
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


def load_gesis():
    gesis = pd.read_csv("../data/gesis_processed.csv")
    gbs = pd.read_csv("../data/gbs_processed.csv")

    gesis_columns = [
        "Geschlecht",
        "Geburtsjahr",
        "Geburtsland",
        "Nationalitaet",
        "Familienstand",
        "Hoechster Bildungsabschluss",
        "Berufliche Ausbildung",
        "Erwerbstaetigkeit",
        "Nettoeinkommen Selbst",
        "Zufriedenheit Wahlergebnis",
        "Gesellig",
        "Andere kritisieren",
        "Gruendlich",
        "Nervoes",
        "Phantasievoll",
        "Berufsgruppe",
        "Wahlteilnahme",
        "BRS6",
    ]

    gbs["label"] = 1
    gesis["label"] = 0

    gesis_gbs = pd.concat([gbs, gesis], ignore_index=True)
    return gesis_gbs, gesis_columns, None


def load_artificial_data():
    artificial_data_path = "../data/artificial.csv"
    artificial = pd.read_csv(artificial_data_path)
    columns = artificial.filter(like='x').columns
    return artificial, columns, None


def load_census_data():
    census_bias = "Marital Status_Married"
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
        "../data/Census_Income/adult.data", names=columns, na_values=["-1", -1, " ?"]
    )
    df, preprocessed_columns = preprocess_census(df, census_bias)
    return df, preprocessed_columns, census_bias, None


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
    ctg = ["Workclass", "Marital Status", "Race", "Occupation", "Country"]
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
        census_bias,
    ]
    for m in meta:
        if m in census_columns:
            census_columns.remove(m)
    df_copy = df.copy()
    df_positive_class = df_copy[(df_copy[census_bias] == 1)].copy()
    df_negative_class = df_copy[(df_copy[census_bias] == 0)].copy()

    rep_fraction = 0.12
    bias_fraction = 0.07
    negative_normal = len(df_negative_class)
    positive_normal = len(df_positive_class)

    rep = pd.concat(
        [
            df_negative_class.head(int(negative_normal * 0.2)),
            df_positive_class.head(int(positive_normal * 0.2)),
        ],
        ignore_index=True,
    )
    nonrep_more_negative_class = pd.concat(
        [
            df_negative_class.tail(int(negative_normal * rep_fraction)),
            df_positive_class.tail(int(positive_normal * (bias_fraction))),
        ],
        ignore_index=True,
    )
    rep["label"] = 0
    nonrep_more_negative_class["label"] = 1
    census_nonrep_more_negative_class = pd.concat(
        [rep.copy(deep=True), nonrep_more_negative_class.copy(deep=True)]
    )
    return census_nonrep_more_negative_class, census_columns


def load_dataset(dataset_name):
    if dataset_name == "allensbach":
        return load_allensbach()
    elif dataset_name == "gesis":
        return load_gesis()
    elif dataset_name == "artificial":
        return load_artificial_data()
    elif dataset_name == "census":
        return load_census_data()
