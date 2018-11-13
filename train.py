from __future__ import division
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from argparse import ArgumentParser
from sklearn.externals import joblib
from pandas import read_csv
from pprint import pprint
from scipy import *


import os
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def create_parser():
    usage = "python3 train.py [data] [cases] [name]"

    parser = ArgumentParser(usage=usage)

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store")
    parser.add_argument("data", help="filepath of cleaned feature training data",
                    action="store")
    parser.add_argument("cases", help="filepath of labeled cases data", action="store")
    parser.add_argument("name", help="name of rf file to save", action="store")

    args = parser.parse_args()

    return args

def train_rf(args):
    data = read_csv(args.data)
    cases = read_csv(args.cases)

    data['total_cases'] = cases['total_cases']

    for counter in range(len(data)):
        if data.at[counter,'city'] == "sj":
            data.at[counter,'city'] = 0
        else:
            data.at[counter,'city'] = 1

    #drop week_start_date since it's pretty much the same as weekofyear
    data.drop(['week_start_date'], axis=1, inplace=True)

    # drop some features for experimenting
    # ex: 'city','week_start_date','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw'
    data.drop(['year'], axis=1, inplace=True)

    # data.drop(['reanalysis_specific_humidity_g_per_kg'], axis=1, inplace=True)
    # data.drop(['reanalysis_max_air_temp_k'], axis=1, inplace=True)
    # data.drop(['station_diur_temp_rng_c'], axis=1, inplace=True)
    data.drop(['station_avg_temp_c'], axis=1, inplace=True)
    data.drop(['precipitation_amt_mm'], axis=1, inplace=True)
    data.drop(['station_min_temp_c'], axis=1, inplace=True)
    data.drop(['reanalysis_sat_precip_amt_mm'], axis=1, inplace=True)
    data.drop(['reanalysis_min_air_temp_k'], axis=1, inplace=True)
    data.drop(['reanalysis_relative_humidity_percent'], axis=1, inplace=True)
    data.drop(['ndvi_nw'], axis=1, inplace=True)
    data.drop(['reanalysis_avg_temp_k'], axis=1, inplace=True)


    # randomly determine if row is training or testing and separate the data
    data['is_train'] = np.random.uniform(0, 1, len(data)) <= .8
    train, test = data[data['is_train'] == True], data[data['is_train'] == False]

    # drop the training column since we don't need it anymore
    train = train.drop('is_train', axis=1)
    test = test.drop('is_train', axis=1)

    regr = RandomForestRegressor(n_jobs=4, n_estimators=500, oob_score=True, max_depth=200)
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
    # print("\nAccuracy: {}".format(100*(correct/len(test))))
    print("Average difference: {}".format(total_diff/len(test)))


    # prints feature ranking
    print("\nFeature ranking:")

    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_],
    axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    for f in range(testnocases.shape[1]):
        if (importances[indices[f]] > 0.005):
            print("%d. feature %s (%f)" % (f + 1, testnocases.columns.values[indices[f]], importances[indices[f]]))

    joblib.dump(regr, args.name)

if __name__ == "__main__":
    args = create_parser()
    train_rf(args)
