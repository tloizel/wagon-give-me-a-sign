from load_from_bq import load_from_bq
import pandas as pd
from sklearn.model_selection import train_test_split
from data_extraction import get_coordinates
import ipdb



df = load_from_bq()

def balance_df(df):
    if len(df.groupby('target')['target'].count().unique()) > 1:
        print("Not balanced")
        print(df.groupby('target')['target'].count().unique())
        min_ = min(df.groupby('target')['target'].count().unique())
        balanced_df = pd.DataFrame()
        for letter in df['target'].unique():
            sub_df = df[df['target']==letter]
            selected_row = sub_df.sample(n=min_)
            balanced_df = pd.concat([balanced_df,selected_row])
    else:
        print("Balanced")
        print(df.groupby('target')['target'].count().unique())
        balanced_df = df

    print(balanced_df.shape)
    return balanced_df

def shuffle_targets(df):
    shuffled_df = pd.DataFrame()
    for letter in df['target'].unique():
        sub_df = df[df['target']==letter]
        sub_df = sub_df.sample(frac=1)
        shuffled_df = pd.concat([shuffled_df,sub_df])
        #shuffled_df = shuffled_df.reset_index(drop=True)

    return shuffled_df

def train_test_df(df, test_size=0.3, random_state=42):

    df = df.reset_index(drop=True)

    X_train_df = pd.DataFrame()
    X_test_df = pd.DataFrame()
    y_train_df = pd.DataFrame()
    y_test_df = pd.DataFrame()

    for letter in df['target'].unique():
        sub_df = df[df['target']==letter]
        X = sub_df.drop('target', axis=1)
        y = sub_df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train_df = pd.concat([X_train_df, X_train])
        X_test_df = pd.concat([X_test_df, X_test])
        y_train_df = pd.concat([y_train_df, y_train])
        y_test_df = pd.concat([y_test_df, y_test])

    return X_train_df, X_test_df, y_train_df, y_test_df

def preproc(df, test_size=0.3, random_state=42):
    """
    Remove duplicates
    Remove the first three colmns x_0, y_0, z_0
    Balance dataset if needed
    Return X_train_df, X_test_df, y_train_df, y_test_df
    """

    # Remove duplicates
    df = df.drop_duplicates()
    df = df.dropna()

    # Remove the first three columns
    if 'x_0' in df.columns:
        df = df.drop(['x_0','y_0','z_0'], axis=1)

    # check balance and balance it
    balanced_df = balance_df(df)

    # # shuffle par target
    # shuffled_df = shuffle_targets(balanced_df)
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_df(balanced_df, test_size=test_size, random_state=random_state)

    return X_train_df, X_test_df, y_train_df, y_test_df


def preproc_predict(image, processed_hand_dict):
    coords = get_coordinates(image, processed_hand_dict)
    if not coords:
        return None
    else :
        coords = pd.DataFrame(coords, index=[0])
        coords = coords.drop(['x_0','y_0','z_0'], axis=1)
        return coords
