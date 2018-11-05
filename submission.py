import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from pandas import read_csv
from pprint import pprint
from scipy import *

if __name__ == "__main__":
    regrPath = input("Input path to random forest to load: ")
    testPath = input("Input path to testing features: ")

    regr = joblib.load(regrPath)
    test = read_csv(testPath)
    labels = read_csv("../data/given_data/submission_format.csv")

    # change city into numerical data
    for counter in range(len(test)):
        if test.at[counter,'city'] == "sj":
            test.at[counter,'city'] = 0
        else:
            test.at[counter,'city'] = 1

    test.drop(['week_start_date'], axis=1, inplace=True)

    print('\nPredicting')

    labels['total_cases'] = regr.predict(test)
    labels.total_cases.round()
    labels['total_cases'] = labels['total_cases'].astype(int)

    labels.to_csv('../data/submission.csv', sep = ',', encoding = 'utf-8', index = False)
