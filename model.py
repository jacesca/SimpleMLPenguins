import pandas as pd

# Reading data
df = pd.read_csv('penguins.csv')
print('\nData sample: ')
print(df.head(10))

print('\nDate ')
# Preprocessing data with scikit-learn
