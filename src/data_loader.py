# data_loader.py
import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_credit_card_dataset():
    # fetch dataset
    default_of_credit_card_clients = fetch_ucirepo(id=350)

    # data (as pandas dataframes)
    X = default_of_credit_card_clients.data.features
    y = default_of_credit_card_clients.data.targets
    df = pd.concat([X, y], axis=1)

    # metadata
    print(default_of_credit_card_clients.metadata)

    # variable information
    print(default_of_credit_card_clients.variables)

    return df
