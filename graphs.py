import numpy as np
from pandas import read_csv
from pprint import pprint
from scipy import *

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

train_features = read_csv('data/cleaned_train_data.csv', index_col=[0,1,2])
train_labels = read_csv('data/given_data/dengue_labels_train.csv', index_col=[0,1,2])

sj_train_features = train_features.loc['sj']
sj_train_labels = train_labels.loc['sj']

iq_train_features = train_features.loc['iq']
iq_train_labels = train_labels.loc['iq']

sj_train_features['total_cases'] = sj_train_labels.total_cases
iq_train_features['total_cases'] = iq_train_labels.total_cases

sj_correlations = sj_train_features.corr()
iq_correlations = iq_train_features.corr()

sj_corr_heat = sns.heatmap(sj_correlations)
plt.title('San Juan Variable Correlations After Interpolation')
plt.show()
