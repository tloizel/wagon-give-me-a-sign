from load_from_bq import load_from_bq
import pandas as pd
from sklearn.model_selection import train_test_split


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

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for letter in df['target'].unique():
        sub_df = df[df['target']==letter]
        train_data, test_data = train_test_split(sub_df, test_size=test_size, random_state=random_state)
        train_df = pd.concat([train_df,train_data])
        test_df = pd.concat([test_df,test_data])

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df

def preproc(df, test_size=0.3, random_state=42):

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

    train, test = train_test_df(balanced_df, test_size=test_size, random_state=random_state)

    return train, test
