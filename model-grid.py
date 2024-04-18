"""
This pipeline prepare the pipe using make_pipeline and then feed the GridSearch
"""

import pandas as pd
import numpy as np
import random
# import tensorflow as tf

from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# Setting the seed to make the process reproducible
np.random.seed(42)
random.seed(42)
# tf.random.set_seed(42)

# Reading data
print("""
-----------------
Reading the data:
-----------------
""")
df = pd.read_csv('penguins.csv')
print('(1). Get an overview of the data')
print(df.head(10))

print('\n\n\n(2). Check missing values: ')
print(df.info())

print('\n\n\n(3). Look at rows containing any missing value')
print(df[df.isna().any(axis=1)])

print('\n\n\n(4). Look unique values in island and in sex columns')
print('island:', df['island'].unique())
print('sex:', df['sex'].unique())

print('\n\n\n(5). Look unique values in the target" species column')
print('species:', df['species'].unique())

# Preprocessing data with scikit-learn
print("""\n\n\n
-----------------
Preprocessing the data:
-----------------
""")
print('(1). Remove rows with too little information')
df = df[df.isna().sum(axis=1) < 2]
print(df.head(10))

print('\n\n\n(2). Separate the data into x and y')
y = df['species']
x = df.drop('species', axis=1)

print('\n\n\n(3). Separate the data into training and testing set')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

print('\n\n\n(4). Encode the target column: species')
y_origin = y_train
labelenc = LabelEncoder()
y_train = labelenc.fit_transform(y_train)
y_decoded = labelenc.inverse_transform(y_train)  # To Decode

temp_df = pd.DataFrame({})
temp_df['target'] = y_origin
temp_df['y encoded'] = y_train
temp_df['y decoded'] = y_decoded
print(temp_df.head(10))

# Modeling
print("""\n\n\n
-----------------
Modeling:
-----------------
""")
print('\n\n\n(1). Build the pipeline')
col_encoder = make_column_transformer((OneHotEncoder(), ['island', 'sex']),
                                      remainder='passthrough')
pipe = make_pipeline(
    col_encoder,
    SimpleImputer(strategy='most_frequent'),
    StandardScaler(),
    KNeighborsClassifier()
)

print('\n\n\n(2). Tuning Hyperparameters')
param_grid = {'kneighborsclassifier__n_neighbors': [1, 3, 5, 7],
              'kneighborsclassifier__weights': ['distance', 'uniform'],
              'kneighborsclassifier__p': [1, 2]}
grid_search = GridSearchCV(estimator=pipe,
                           param_grid=param_grid,
                           cv=3)

print('\n\n\n(3). Train the model')
grid_search.fit(x_train, y_train)
print('Best estimator:', grid_search.best_estimator_)
print('Best score:', grid_search.best_score_)

# Evaluating
print("""\n\n\n
-----------------
Evaluating:
-----------------
""")
print('\n\n\n(1). Make predictions')
y_pred = grid_search.predict(x_train)  # retrieve encoded predictions
y_pred_decoded = labelenc.inverse_transform(y_pred)
temp_df = pd.DataFrame({})
temp_df['target'] = y_origin
temp_df['y train'] = y_train
temp_df['y pred encoded'] = y_pred
temp_df['y pred decoded'] = y_pred_decoded
print(temp_df)

print('\n\n\n(2). Evaluate the model')
y_origin_test = y_test
y_test = labelenc.transform(y_test)
print('Score:', grid_search.score(x_test, y_test))

y_origin = y
y = labelenc.transform(y)
scores = cross_val_score(grid_search, x, y, cv=3)
print('CrossVal score:', scores.mean())
