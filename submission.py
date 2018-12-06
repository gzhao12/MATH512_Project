import numpy as np
from sklearn.ensemble import RandomForestRegressor
from argparse import ArgumentParser
from sklearn.externals import joblib
import pandas as pd
from pprint import pprint
from scipy import *
from warnings import filterwarnings

import os

filterwarnings('ignore')

def create_parser():
    usage = "python3 clean.py [regr] [test] [save]"
    parser = ArgumentParser(usage=usage)

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store")
    parser.add_argument("regr", help="filepath of regression you want to use",
                    action="store")
    parser.add_argument("test", help="filepath of test data", action="store")
    parser.add_argument("save", help="filepath and name to save csv file", action="store")

    args = parser.parse_args()

    return args

def create_submission(args):
    sj_regr = joblib.load(args.regr + "_sj")
    iq_regr = joblib.load(args.regr + "_iq")
    test = pd.read_csv(args.test, index_col=[0,1])
    labels = pd.read_csv("data/given_data/submission_format.csv")

    # drop week_start_date since its pretty much the same as weekofyear
    test.drop(['week_start_date'], axis=1, inplace=True)

    sj_test = test.loc['sj']
    iq_test = test.loc['iq']

    # drop some features for experimenting
    # ex: 'city','week_start_date','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw'

    sj_test.drop(['station_diur_temp_rng_c'], axis=1, inplace=True)
    sj_test.drop(['precipitation_amt_mm'], axis=1, inplace=True)
    sj_test.drop(['station_precip_mm'], axis=1, inplace=True)
    sj_test.drop(['reanalysis_sat_precip_amt_mm'], axis=1, inplace=True)

    iq_test.drop(['ndvi_se'], axis=1, inplace=True)
    iq_test.drop(['ndvi_ne'], axis=1, inplace=True)
    iq_test.drop(['ndvi_sw'], axis=1, inplace=True)
    iq_test.drop(['ndvi_nw'], axis=1, inplace=True)
    iq_test.drop(['station_precip_mm'], axis=1, inplace=True)

    # test.drop(['reanalysis_specific_humidity_g_per_kg'], axis=1, inplace=True)
    # test.drop(['reanalysis_max_air_temp_k'], axis=1, inplace=True)
    # test.drop(['station_diur_temp_rng_c'], axis=1, inplace=True)

    # test.drop(['year'], axis=1, inplace=True)
    #
    # test.drop(['station_avg_temp_c'], axis=1, inplace=True)
    # test.drop(['precipitation_amt_mm'], axis=1, inplace=True)
    # test.drop(['station_min_temp_c'], axis=1, inplace=True)
    # test.drop(['reanalysis_sat_precip_amt_mm'], axis=1, inplace=True)
    # test.drop(['reanalysis_min_air_temp_k'], axis=1, inplace=True)
    # test.drop(['reanalysis_relative_humidity_percent'], axis=1, inplace=True)
    # test.drop(['ndvi_nw'], axis=1, inplace=True)
    # test.drop(['reanalysis_avg_temp_k'], axis=1, inplace=True)

    print('\nPredicting')

    sj_cases = pd.Series(sj_regr.predict(sj_test))
    iq_cases = pd.Series(iq_regr.predict(iq_test))

    predictions = pd.concat([sj_cases, iq_cases], ignore_index=True)
    predictions.round()
    predictions = predictions.astype(int)

    labels['total_cases'] = predictions.values

    # print(labels)
    labels.to_csv(args.save, sep = ',', encoding = 'utf-8', index = False)


if __name__ == "__main__":

    args = create_parser()
    create_submission(args)
