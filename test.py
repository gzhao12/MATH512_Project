import math
import random
import numpy as np
from pandas import read_csv
from argparse import ArgumentParser

import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def create_parser():
    usage = "python3 clean.py [data]"
    parser = ArgumentParser(usage=usage)

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store")
    parser.add_argument("data", help="filepath of uncleaned feature training data",
                    action="store")

    args = parser.parse_args()

    return args

def print_file(args):
    data = read_csv(args.data)
    print(data)
if __name__ == "__main__":
    args = create_parser()
    print_file(args)
