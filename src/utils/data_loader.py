import pandas as pd
import pathlib
from folktables import ACSDataSource, generate_categories, BasicProblem, adult_filter
import numpy as np

file_path = pathlib.Path(__file__).parent


def load_gbs_allensbach():
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
    columns = artificial.filter(like="x").columns.tolist()
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


def load_dataset(dataset_name, bias_variable):
    if dataset_name == "gbs_allensbach":
        return load_gbs_allensbach()
    elif dataset_name == "artificial":
        return load_artificial_data()
    elif dataset_name == "census":
        return load_census_data(bias_variable)
    elif dataset_name == "folktables":
        return load_folktables_data()
    elif dataset_name == "mrs_census":
        return load_mrs_census_data(bias_variable)
    elif dataset_name == "breast_cancer":
        return load_brast_cancer_data()
    else:
        print("No valid data set name given!")
        exit()


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
    usa_data = data_source.get_data(states=["CA"], download=True)
    definition_df = data_source.get_definitions(download=True)
    categories = generate_categories(
        features=ACSIncomeNew.features, definition_df=definition_df
    )

    usa_features, us_labels, _ = ACSIncomeNew.df_to_pandas(
        usa_data, categories=categories, dummies=True
    )

    columns = usa_features.columns
    usa_features["Income"] = us_labels
    usa_features = usa_features.dropna()

    return usa_features, columns


ACSIncomeNew = BasicProblem(
    features=[
        "AGEP",
        "COW",
        "SCHL",
        "MAR",
        "RELP",
        "WKHP",
        "SEX",
        "RAC1P",
    ],
    target="PINCP",
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)


def load_mrs_census_data(census_bias):
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


def sample_mrs_census(bias_type, df, columns, bias_variable, folktables=False):
    rep_fraction = 0.12
    bias_fraction = 0.05
    R_fraction = 0.2

    if not folktables:
        negative_normal = len(df[(df[bias_variable] == 0)])
        positive_normal = len(df[(df[bias_variable] == 1)])
        df_positive_class = df[(df[bias_variable] == 1)]
        df_negative_class = df[(df[bias_variable] == 0)]
    else:
        negative_normal = len(df[(df[bias_variable] <= 50000)])
        positive_normal = len(df[(df[bias_variable] > 50000)])
        df_positive_class = df[(df[bias_variable] <= 50000)]
        df_negative_class = df[(df[bias_variable] > 50000)]
        rep_fraction *= 0.1
        R_fraction *= 0.1
        bias_fraction *= 0.1

    scaled_R = pd.concat(
        [
            df_negative_class.sample(int(negative_normal * R_fraction)),
            df_positive_class.sample(int(positive_normal * R_fraction)),
        ],
        ignore_index=True,
    )
    if bias_type == "less_positive_class":
        scaled_N = pd.concat(
            [
                df_negative_class.sample(int(negative_normal * rep_fraction)),
                df_positive_class.sample(
                    int(positive_normal * (rep_fraction - bias_fraction))
                ),
            ],
            ignore_index=True,
        )
    elif bias_type == "less_negative_class":
        scaled_N = pd.concat(
            [
                df_negative_class.sample(
                    int(negative_normal * (rep_fraction - bias_fraction))
                ),
                df_positive_class.sample(int(positive_normal * rep_fraction)),
            ],
            ignore_index=True,
        )
    elif bias_type == "mean_difference":
        mean_sample = df[columns].mean().values
        sample_weights = [
            np.exp(-(1 / 20) * (np.linalg.norm(sample - mean_sample) ** 2))
            for sample in df[columns].values
        ]
        scaled_N = df.sample(weights=sample_weights, frac=0.1)
        scaled_N = scaled_N.reset_index(drop=True)
    elif bias_type == "high_bias":
        scaled_N = pd.concat(
            [
                df_negative_class.sample(int(negative_normal * 0.03)),
                df_positive_class.sample(int(positive_normal * 0.07)),
            ],
            ignore_index=True,
        )
    else:
        scaled_N = pd.concat(
            [
                df_negative_class.sample(int(negative_normal * rep_fraction)),
                df_positive_class.sample(int(positive_normal * rep_fraction)),
            ],
            ignore_index=True,
        )

    scaled_R["label"] = 0
    scaled_N["label"] = 1

    return scaled_N, scaled_R


breast_cancer_names = [
    "sample_code_number",
    "clump_thickness",
    "uniformity_of_cell_size",
    "uniformity_of_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class",
]


def load_brast_cancer_data():
    df = pd.read_csv(
        f"{file_path}/../../data/breast_cancer/breast-cancer-wisconsin.data",
        na_values=["?"],
        names=breast_cancer_names,
    )
    df = df.dropna()
    replace_dict = {2: 1, 4: 0}
    df["class"] = df["class"].replace(replace_dict)
    df = df.drop(columns=["sample_code_number"])
    columns = df.drop(columns=["class"]).columns
    df = df.reset_index()

    return df, columns


def sample_breast_cancer(bias_variable, df, bias_type, columns):
    if bias_type == "mean_difference":
        mean_sample = df[columns].mean().values
        sample_weights = [
            np.exp(-(1 / 20) * (np.linalg.norm(sample - mean_sample) ** 2))
            for sample in df[columns].values
        ]
        N = df.sample(weights=sample_weights, frac=0.25)
        R = df[~df.index.isin(N.index)]

    else:
        train_size = int(0.25 * len(df))
        parameter = df[bias_variable]
        weights = np.where(parameter <= 5, 0.2, 0.8)
        weights = weights / sum(weights)
        train_indices = np.random.choice(len(df), train_size, replace=False, p=weights)
        N = df.iloc[train_indices]
        R = df.drop(train_indices)

    N = N.copy().reset_index()
    R= R.copy().reset_index()
    R["label"] = 0
    N["label"] = 1

    return N, R
