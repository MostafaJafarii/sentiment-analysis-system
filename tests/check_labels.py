import pandas as pd

y_train = pd.read_csv("data/processed/y_train.csv")

print(y_train["sentiment"].value_counts())

print()

print(y_train["sentiment"].unique())