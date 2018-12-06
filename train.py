import numpy as np
from sklearn.ensemble import RandomForestRegressor
from argparse import ArgumentParser
from sklearn.externals import joblib
from pandas import read_csv
from pprint import pprint
from warnings import filterwarnings
from scipy import *

import os

filterwarnings('ignore')

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
    data = read_csv(args.data, index_col=[0,1])
    cases = read_csv(args.cases, index_col=[0,1])

    sj_train_features = data.loc['sj']
    sj_train_labels = cases.loc['sj']

    iq_train_features = data.loc['iq']
    iq_train_labels = cases.loc['iq']

    sj_train_features['total_cases'] = sj_train_labels['total_cases']
    iq_train_features['total_cases'] = iq_train_labels['total_cases']

    #drop week_start_date since it's pretty much the same as weekofyear
    sj_train_features.drop(['week_start_date'], axis=1, inplace=True)
    iq_train_features.drop(['week_start_date'], axis=1, inplace=True)

    # drop some features for experimenting
    # ex: 'city','week_start_date','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw'

    sj_train_features.drop(['station_diur_temp_rng_c'], axis=1, inplace=True)
    sj_train_features.drop(['precipitation_amt_mm'], axis=1, inplace=True)
    sj_train_features.drop(['station_precip_mm'], axis=1, inplace=True)
    # sj_train_features.drop(['ndvi_se'], axis=1, inplace=True)
    # sj_train_features.drop(['ndvi_ne'], axis=1, inplace=True)
    # sj_train_features.drop(['ndvi_sw'], axis=1, inplace=True)
    # sj_train_features.drop(['ndvi_nw'], axis=1, inplace=True)
    sj_train_features.drop(['reanalysis_sat_precip_amt_mm'], axis=1, inplace=True)

    # data.drop(['reanalysis_specific_humidity_g_per_kg'], axis=1, inplace=True)
    # data.drop(['reanalysis_max_air_temp_k'], axis=1, inplace=True)
    # data.drop(['station_avg_temp_c'], axis=1, inplace=True)
    # data.drop(['station_min_temp_c'], axis=1, inplace=True)
    # data.drop(['reanalysis_min_air_temp_k'], axis=1, inplace=True)
    # data.drop(['reanalysis_relative_humidity_percent'], axis=1, inplace=True)
    # data.drop(['ndvi_nw'], axis=1, inplace=True)
    # data.drop(['reanalysis_avg_temp_k'], axis=1, inplace=True)

    iq_train_features.drop(['ndvi_se'], axis=1, inplace=True)
    iq_train_features.drop(['ndvi_ne'], axis=1, inplace=True)
    iq_train_features.drop(['ndvi_sw'], axis=1, inplace=True)
    iq_train_features.drop(['ndvi_nw'], axis=1, inplace=True)
    iq_train_features.drop(['station_precip_mm'], axis=1, inplace=True)

    # randomly determine if row is training or testing and separate the data
    sj_train_features['is_train'] = np.random.uniform(0, 1, len(sj_train_features)) <= .9
    iq_train_features['is_train'] = np.random.uniform(0, 1, len(iq_train_features)) <= .9

    sj_train, sj_test = sj_train_features[sj_train_features['is_train'] == True],  sj_train_features[sj_train_features['is_train'] == False]
    iq_train, iq_test = iq_train_features[iq_train_features['is_train'] == True], iq_train_features[iq_train_features['is_train'] == False]


    # drop the training column since we don't need it anymore
    sj_train = sj_train.drop(['is_train'], axis=1)
    sj_test = sj_test.drop(['is_train'], axis=1)

    iq_train = iq_train.drop(['is_train'], axis=1)
    iq_test = iq_test.drop(['is_train'], axis=1)


    sj_regr = RandomForestRegressor(n_jobs=4, n_estimators=500, max_depth=72, oob_score=True)
    iq_regr = RandomForestRegressor(n_jobs=4, n_estimators=100, max_depth=36, oob_score=True)

    sj_y = sj_train['total_cases']
    iq_y = iq_train['total_cases']

    print('\nTraining')
    sj_regr.fit(sj_train.drop(['total_cases'], axis=1), sj_y)
    iq_regr.fit(iq_train.drop(['total_cases'], axis=1), iq_y)

    sj_testnocases = sj_test.drop('total_cases', axis=1)
    iq_testnocases = iq_test.drop('total_cases', axis=1)

    print('\nPredicting')

    # evaluate our results on the test set and round/convert to int.
    sj_test['prediction'] = sj_regr.predict(sj_testnocases)
    sj_test.prediction.round()
    sj_test['prediction'] = sj_test['prediction'].astype(int)

    iq_test['prediction'] = iq_regr.predict(iq_testnocases)
    iq_test.prediction.round()
    iq_test['prediction'] = iq_test['prediction'].astype(int)

    sj_results = sj_test.groupby(['total_cases', 'prediction'])
    sj_resultsagg = sj_results.size()
    print("San Juan:")
    print(sj_resultsagg)

    iq_results = iq_test.groupby(['total_cases', 'prediction'])
    iq_resultsagg = iq_results.size()
    print("Iquitos:")
    print(iq_resultsagg)

    # prints percentage of correct predictions and average difference
    sj_correct = 0
    sj_total_diff = 0
    for counter in range(len(sj_test)):
        if sj_test['total_cases'].iat[counter] != sj_test['prediction'].iat[counter]:
            sj_total_diff += abs(sj_test['total_cases'].iat[counter] - sj_test['prediction'].iat[counter])
        else:
            sj_correct += 1

    # print("\nAccuracy: {}".format(100*(correct/len(test))))
    print("San Juan average difference: {}".format(sj_total_diff/len(sj_test)))

    # prints percentage of correct predictions and average difference
    iq_correct = 0
    iq_total_diff = 0
    for counter in range(len(iq_test)):
        if iq_test['total_cases'].iat[counter] != iq_test['prediction'].iat[counter]:
            iq_total_diff += abs(iq_test['total_cases'].iat[counter] - iq_test['prediction'].iat[counter])
        else:
            iq_correct += 1

    # print("\nAccuracy: {}".format(100*(correct/len(test))))
    print("Iquitos average difference: {}".format(iq_total_diff/len(iq_test)))


    # prints feature ranking
    # print("\nFeature ranking:")
    #
    # importances = sj_regr.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in sj_regr.estimators_],
    # axis=0)
    # indices = np.argsort(importances)[::-1]
    #
    # # Print the feature ranking
    # for f in range(sj_testnocases.shape[1]):
    #     if (importances[indices[f]] > 0.005):
    #         print("%d. feature %s (%f)" % (f + 1, sj_testnocases.columns.values[indices[f]], importances[indices[f]]))

    # joblib.dump(sj_regr, args.name + "_sj")
    # joblib.dump(iq_regr, args.name + "_iq")

if __name__ == "__main__":
    args = create_parser()
    train_rf(args)
