from scipy import *
from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import linalg
from pprint import pprint

features_training = read_csv('data/dengue_features_train.csv')
labels_training = read_csv('data/dengue_labels_train.csv')

'''
starts at week of the year from features training file and then moves right
each column. Thus year and city and not included.
'''
# TODO add a for loop and replace week of year with the counter to iterate
# for each column in the csv file using iloc
y = double(array(labels_training['total_cases']))
p1 = double(array(features_training['weekofyear']))

plt.figure()
plt.plot(p1[:7],y[:7],'*')
plt.grid()
plt.show()
