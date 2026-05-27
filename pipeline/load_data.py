import pandas as pd

from pipeline.config import DATA_PATH


def load_dataset():

    df = pd.read_csv(DATA_PATH, index_col=0)

    df = df.dropna().reset_index(drop=True)

    return df