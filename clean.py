from scipy.optimize import curve_fit
from scipy import linalg
from pprint import pprint
from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import math
import random
import scipy
import os
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def truncate(number, digits) -> float:
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper

def create_parser():
    usage = "python3 clean.py [data] [save]"
    parser = ArgumentParser(usage=usage)

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store")
    parser.add_argument("data", help="filepath of uncleaned feature training data",
                    action="store")
    parser.add_argument("save", help="filepath and name to save csv file", action="store")

    args = parser.parse_args()

    return args

# CHANGE THIS METHOD BASED ON THE LINEAR REGRESSION CODE
def fill(args):
    data = pd.read_csv(args.data)

    for counter1, row in data.iterrows():
        for counter2, col in data.iteritems():
            if(data.isnull().at[counter1, counter2]):
                week_arr = []
                data_arr = []
                nulls_arr = []
                counter_arr = []
                arr_iter = 0
                std_dev = 0
                tmpCounter = counter1 - 3
                for tmpCounter in range(tmpCounter, tmpCounter + 3):
                    tmp = [data.iat[tmpCounter, 2], 1]
                    week_arr.append(tmp)
                    data_arr.append(data.at[tmpCounter, counter2])

                tmpCounter += 1

                while(len(week_arr) < 6):
                    if(data.isnull().at[tmpCounter, counter2]):
                        tmp = [data.iat[tmpCounter, 2], 1]
                        nulls_arr.append(tmp)
                        counter_arr.append(tmpCounter)
                        tmpCounter += 1
                    else:
                        tmp = [data.iat[tmpCounter, 2], 1]
                        week_arr.append(tmp)
                        data_arr.append(data.at[tmpCounter, counter2])
                        tmpCounter += 1

                lin_reg = LinearRegression().fit(week_arr, data_arr)
                predictions = lin_reg.predict(week_arr)

                if(args.verbose):
                    print(data_arr)
                    print(predictions)

                for value in data_arr:
                    std_dev += ((value - predictions[arr_iter]) ** 2)
                    arr_iter += 1

                std_dev = (std_dev/6) ** .5

                if(args.verbose):
                    print(std_dev)

                arr_iter = 0

                for counter in counter_arr:
                    prediction = lin_reg.predict([nulls_arr[arr_iter]])
                    rand = np.random.normal(0, std_dev)
                    data.at[counter, counter2] = prediction + rand
                    if(args.verbose):
                        print(rand)
                        print(data.at[counter, counter2])
                    arr_iter += 1

    data.to_csv(args.save, sep = ',', encoding = 'utf-8', index = False)
    print("Saved to", args.save, "!")

if __name__ == "__main__":
    args = create_parser()
    fill(args)


# '''
# find rows with more null values than specified and removes them from the data set
# '''
# def remove_null_rows():
#     features_training = pd.read_csv('data/given_data/dengue_features_train.csv')
#     cases = pd.read_csv('data/given_data/dengue_labels_train.csv')
#
#     null_rows = 0 #rows with 4 or more nans
#     for counter1, row in features_training.iterrows():
#         null_counter = 0 #counts values with NaN
#         for counter2, col in features_training.iteritems():
#             # print (features_training.at[counter1, counter2])
#             if(features_training.isnull().at[counter1, counter2]):
#                 null_counter += 1
#         if(null_counter >= 4):
#             print("Removing row: ", counter1)
#             features_training.drop(index = counter1, inplace = True)
#             cases.drop(index = counter1, inplace = True)
#             null_rows += 1
#
#     print("There were ", null_rows, " rows with 4 or more nans.")
#     features_training.to_csv('cleaned_data.csv', sep = ',', encoding = 'utf-8', index = False)
#     cases.to_csv('cleaned_data_cases.csv', sep = ',', encoding = 'utf-8', index = False)
#
# def remove_any_null_row():
#     features_training = pd.read_csv('data/given_data/dengue_features_train.csv')
#     cases = pd.read_csv('data/given_data/dengue_labels_train.csv')
#
#     null_rows = 0 #rows with 4 or more nans
#     for counter1, row in features_training.iterrows():
#         null_counter = 0 #counts values with NaN
#         for counter2, col in features_training.iteritems():
#             # print (features_training.at[counter1, counter2])
#             if(features_training.isnull().at[counter1, counter2]):
#                 null_counter += 1
#         if(null_counter > 0):
#             print("Removing row: ", counter1)
#             features_training.drop(index = counter1, inplace = True)
#             cases.drop(index = counter1, inplace = True)
#             null_rows += 1
#
#     print("There were ", null_rows, " rows with 0 or more nans.")
#     features_training.to_csv('no_nulls.csv', sep = ',', encoding = 'utf-8', index = False)
#     cases.to_csv('no_nulls_cases.csv', sep = ',', encoding = 'utf-8', index = False)



# for counter1, row in data.iterrows():
#     for counter2, col in data.iteritems():
#         avg_diff = 0
#         if(data.isnull().at[counter1, counter2]):
#             for counter3 in range(1,5):
#                 avg_diff += abs(data.at[counter1 - counter3, counter2] - data.at[counter1 - counter3 - 1, counter2])
#             avg_diff /= 4
#             avg_diff = truncate(avg_diff, 6)
#
#             rand = random.randint(1,10)
#
#             if (args.verbose):
#                 print("Random number generated: ", rand)
#
#             noise = truncate(random.uniform(-avg_diff*.05,avg_diff*.05), 6)
#
#             if (args.verbose):
#                 print("Random noise generated: ", noise)
#
#             if(rand%2 == 0):
#                 data.at[counter1, counter2] = data.at[counter1 - 1, counter2] + avg_diff + noise
#                 if (args.verbose):
#                     print("Row: ", counter1, "Col: ", counter2, "has value: ", data.at[counter1, counter2])
#             else:
#                 data.at[counter1, counter2] = data.at[counter1 - 1, counter2] + avg_diff + noise
#                 if (args.verbose):
#                     print("Row: ", counter1, "Col: ", counter2, "has value: ", data.at[counter1, counter2])
