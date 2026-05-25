import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df.drop_duplicates(inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df
