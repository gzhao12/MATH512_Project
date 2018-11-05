from __future__ import division
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from pandas import read_csv
from pprint import pprint
from scipy import *
# from optparse import OptionParser
# import logging
# import os

if __name__ == "__main__":
    dataPath = input("Input the path to the training features: ")
    casesPath = input("Input the path to the training labels: ")
    rfName = input("Input path to save random forest at: ")
    data = read_csv(dataPath)
    cases = read_csv(casesPath)

    # add cases to the dataframe
    data['total_cases'] = cases['total_cases']

    # change city into numerical data
    for counter in range(len(data)):
        if data.at[counter,'city'] == "sj":
            data.at[counter,'city'] = 0
        else:
            data.at[counter,'city'] = 1

    # drop some features for experimenting
    # ex: 'city','week_start_date','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw'
    data.drop(['week_start_date'], axis=1, inplace=True)

    # randomly determine if row is training or testing and separate the data
    data['is_train'] = np.random.uniform(0, 1, len(data)) <= .9
    train, test = data[data['is_train'] == True], data[data['is_train'] == False]

    # drop the training column since we don't need it anymore
    train = train.drop('is_train', axis=1)
    test = test.drop('is_train', axis=1)

    regr = RandomForestRegressor(n_jobs=4, n_estimators=50, oob_score=True)
    y = train['total_cases']

    print('\nTraining')
    regr.fit(train.drop('total_cases', axis=1), y)

    testnocases = test.drop('total_cases', axis=1)

    print('\nPredicting')

    # evaluate our results on the test set and round/convert to int.
    test['prediction'] = regr.predict(testnocases)
    test.prediction.round()
    test['prediction'] = test['prediction'].astype(int)

    results = test.groupby(['total_cases', 'prediction'])
    resultsagg = results.size()
    print(resultsagg)

    # prints percentage of correct predictions and average difference
    correct = 0
    total_diff = 0
    for counter in range(len(test)):
        if test['total_cases'].iat[counter] == test['prediction'].iat[counter]:
            correct += 1
        else:
            total_diff += abs(test['total_cases'].iat[counter] - test['prediction'].iat[counter])
    print("\nAccuracy: {}".format(100*(correct/len(test))))
    print("Average difference: {}".format(total_diff/len(test)))

    # prints feature ranking
    '''
    print("\nFeature ranking:")

    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_],
    axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    for f in range(testnocases.shape[1]):
        if (importances[indices[f]] > 0.005):
            print("%d. feature %s (%f)" % (f + 1, testnocases.columns.values[indices[f]], importances[indices[f]]))
    '''


    joblib.dump(regr, rf)
