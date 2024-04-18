"""
This file only make the data transformation approach
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


# Reading data
print("""
-----------------
Reading the data:
-----------------
""")
df = pd.read_csv('penguins.csv')
print('(1). Looking at the content: ')
print(df.head(10))

print('\n\n\n(2). Checking missing values: ')
print(df.info())

print('\n\n\n(3). Looking at tge rows containing any missing value')
print(df[df.isna().any(axis=1)])

print('\n\n\n(4). Looking unique values in island and in sex columns')
print('island:', df['island'].unique())
print('sex:', df['sex'].unique())

print('\n\n\n(5). Looking unique values in the target" species column')
print('species:', df['species'].unique())

# Separating the data into x and y
y = df['species']
x = df.drop('species', axis=1)

# Preprocessing data with scikit-learn
print("""\n\n\n
-----------------
Preprocessing the data:
-----------------
""")
# 1. Removing rows with too little information
x = x[x.isna().sum(axis=1) < 2]
print('(1). Removing rows with too little information: ')
print(x.head(10))

# 2. Imputting missing values in sex column
imputer = SimpleImputer(strategy='most_frequent')
x['sex'] = imputer.fit_transform(x.sex.values.reshape(-1, 1)).ravel()
x['sex'] = imputer.fit_transform(x[['sex']]).ravel()
print('\n\n\n(2). Imputing mode (most frequent value) in the sex column')
print(x.head(10))

# 3. Imputting missing values in flipper_length_mm and body_mass_g columns
imputer_mean1 = SimpleImputer(strategy='mean')
imputer_mean2 = SimpleImputer(strategy='mean')
x['flipper_length_mm'] = imputer_mean1.fit_transform(
                            x.flipper_length_mm.values.reshape(-1, 1)
                         ).ravel()
x['body_mass_g'] = imputer_mean2.fit_transform(x[['body_mass_g']]).ravel()
print('\n\n\n(3). Imputing mean in flipper_length_mm  and body_mass_g cols')
print(x.head(10))

# 4. Encoding sex and island columns
#    The OrdinalEncoder and OneHotEncoder are usually used to encode features
#    (the X variable).
#    The LabelEncoder is used to encode the target, regardless of whether it
#    is nominal or ordinal.
onehot = OneHotEncoder()
encoded = onehot.fit_transform(x[['island', 'sex']]).toarray()
x.drop(['island', 'sex'], axis=1, inplace=True)
x[onehot.get_feature_names_out()] = encoded
print('\n\n\n(4). Encoding the sex and island columns')
print(x.head(10))

# 5. Encoding the target column: species
labelenc = LabelEncoder()
y = labelenc.fit_transform(y)
print('\n\n\n(5). Encoding the target column: species')
print(y)

# 6. Scaling the data
#    - MinMaxScaler: scales features to a [0,1] range.
#    - MaxAbsScaler: scales features such as the maximum absolute
#                    value is 1 (so the data is guaranteed to be
#                    in a [-1, 1] range).
#    - StandarScalar: standardize features making the mean equal to
#                     0 and variance equal to 1.
#                     **Less sensitive to outliers**
scaler = StandardScaler()
x = scaler.fit_transform(x)
print('\n\n\n(6). Scaling the data')
print(x)
