from load_from_bq import load_from_bq
import pandas as pd


df = load_from_bq()




# Remove duplicates
df = df.drop_duplicates()

# Remove the first three columns
df = df.iloc[:, 3:]

# check balance
# shuffle par target

# split par target
# grouper tous les train et les shuffle
# grouper tous les validation et les shuffle
# grouper tous les test et les shuffle
