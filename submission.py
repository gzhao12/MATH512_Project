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
    regr = joblib.load(args.regr)
    test = read_csv(args.test)
    labels = read_csv("data/given_data/submission_format.csv")

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

    labels.to_csv(args.save, sep = ',', encoding = 'utf-8', index = False)

if __name__ == "__main__":

    args = create_parser()
    create_submission(args)
