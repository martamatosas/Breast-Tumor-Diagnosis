#Load libraries
import pandas as pd
import numpy as np

data = pd.read_csv('breast-cancer-wisconsin-data.csv', header=0)
# Get shape, inspect head and tail and column names
print('The dataset has', data.shape[0], 'rows and', data.shape[1], 'columns')
print(data.head())
print(data.tail())
print('The column names are:', data.columns)
# Inspect structure of the dataframe
print('Structure of the dataframe:')
print(data.info())
# Check for missing values
print('Confirm missing values information:')
print(data.isna().sum())
# Drop column 32 with all nan values and confirm operation
print('Confirm drop operation of missing values:')
data.dropna(axis=1, inplace=True)
print(data.isna().sum())
# Get basic information about the dataset
print('Data types of the features')
print(data.dtypes)
# Column ID can be dropped
data.drop(['id'], axis=1, inplace=True)
# One column is categorical and the rest are all numerical
# The categorical feature is the target: 'diagnosis'
# Inspect the target
print('Unique values and count of the target:')
target_count = data['diagnosis'].value_counts()
print(target_count)
countB = np.sum(data['diagnosis'] == 'B')
countM = np.sum(data['diagnosis'] == 'M')
percentage_benign = (countB/data.shape[0])*100
percentage_malignant = (countM/data.shape[0])*100
print(round(percentage_benign, 2), 'of the samples correspond to a benign diagnosis')
print(round(percentage_malignant, 2), 'of the samples correspond to a malignant diagnosis')
# The percentages are consistent with real life findings
# Data quality: check for values less than zero among the features
X = data.drop(columns='diagnosis')
print('Number of values less than zero:', X.agg(lambda x: sum(x < 0.000000)).sum())
subset_mean = data.iloc[:, 1:11]
subset_se = data.iloc[:, 11:21]
subset_worst = data.iloc[:, 21:31]
print('Summary statistics of the numerical variables in subset "mean":')
print(subset_mean.describe())
print(subset_mean.shape)
print('Summary statistics of the numerical variables in subset "standard error":')
print(subset_se.describe())
print(subset_se.shape)
print('Summary statistics of the numerical variables in subset "worst":')
print(subset_worst.describe())
print(subset_worst.shape)
# Pickle data
data.to_pickle('data')

