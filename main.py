from tqdm import tqdm
import numpy as np
import pandas as pd


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in tqdm(range(len(sequence))):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence.iloc[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return X, y


def add_time_features(df):
    df['Hour sine'] = np.sin(2 * np.pi * df['start_time'].dt.hour / 24)
    df['Hour cosine'] = np.cos(2 * np.pi * df['start_time'].dt.hour / 24)
    df['Day sine'] = np.sin(2 * np.pi * df['start_time'].dt.day / 365.2524)
    df['Day cosine'] = np.cos(2 * np.pi * df['start_time'].dt.day / 365.2524)
    df['Month sine'] = np.sin(2 * np.pi * df['start_time'].dt.month / 12)
    df['Month cosine'] = np.cos(2 * np.pi * df['start_time'].dt.month / 12)
    df['Week sine'] = np.sin(2 * np.pi * df['start_time'].dt.isocalendar().week / 52)
    df['Week cosine'] = np.cos(2 * np.pi * df['start_time'].dt.isocalendar().week / 52)
    df['Year sine'] = np.sin(2 * np.pi * df['start_time'].dt.year / 365.2524)
    df['Year cosine'] = np.cos(2 * np.pi * df['start_time'].dt.year / 365.2524)
    return df

def load_data():
    """
    Loads the data from the given file_name
    """
    df = pd.read_csv("data/no1_train.csv")
    df_validation = pd.read_csv("data/no1_validation.csv")
    df['start_time'] = pd.to_datetime(df['start_time'])
    df.drop(['river'], axis=1, inplace=True)
    df = add_time_features(df)

    column_indcies = ['hydro', 'micro', 'thermal', 'wind', 'total', 'sys_reg', 'flow', 'Hour sine', 'Hour cosine',
                      'Day sine', 'Day cosine', 'Month sine', 'Month cosine', 'Week sine', 'Week cosine', 'Year sine',
                      'Year cosine', 'y']

    train_df = df[column_indcies]
    x_train = train_df.drop(['y'], axis=1)
    y_train = train_df['y']

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train_norm = (x_train - x_train_mean) / x_train_std

    X, y = split_sequence(x_train_norm, 12 * 24 * 7)

    return X, y


if __name__ == '__main__':
    X, y = load_data()
    X = np.array(X)
    print(X.shape)


