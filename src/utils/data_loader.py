import pandas as pd


def load_allensbach():
    allensbach_path = '../data/allensbach_mrs.csv'
    allensbach = pd.read_csv(allensbach_path)
    allensbach.drop(['Unnamed: 0'], axis=1, inplace=True)
    allensbach_columns = ['Alter', 'Berufsgruppe', 'Erwerbstaetigkeit', 'Geschlecht',
                          'Optimismus', 'Pessimismus', 'Schulabschluss', 'woechentlicheArbeitszeit', 'Resilienz']
    return allensbach, allensbach_columns


def load_gesis():
    gesis = pd.read_csv('../data/gesis_processed.csv')
    gbs = pd.read_csv('../data/gbs_processed.csv')

    gesis_columns = ['Geschlecht', 'Geburtsjahr', 'Geburtsland',
                     'Nationalitaet', 'Familienstand', 'Hoechster Bildungsabschluss',
                     'Berufliche Ausbildung', 'Erwerbstaetigkeit', 'Nettoeinkommen Selbst',
                     'Zufriedenheit Wahlergebnis', 'Gesellig', 'Andere kritisieren',
                     'Gruendlich', 'Nervoes', 'Phantasievoll', 'Berufsgruppe', 'Wahlteilnahme', 'BRS6']

    gbs['label'] = 1
    gesis['label'] = 0

    gesis_gbs = pd.concat([gbs, gesis], ignore_index=True)
    return gesis_gbs, gesis_columns


def load_artificial_data():
    artificial_data_path = 'data_propensity/ArtifPopulation.csv'
    artificial = pd.read_csv(artificial_data_path)
    return artificial, artificial.columns


def load_dataset(dataset_name):
    if dataset_name == 'allensbach':
        return load_allensbach()
    elif dataset_name == 'gesis':
        return load_gesis()
    elif dataset_name == 'artificial':
        return load_artificial_data()
