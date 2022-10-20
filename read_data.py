"""
This scripts contains a function to read the UCI Heart Disease data
"""

import pandas as pd
import numpy as np


def read_data(datasets=["processed.cleveland"], separators=[","]):

    datasets = datasets
    separators = separators
    data = []

    for i, dataset in enumerate(datasets):

        data.append(pd.read_csv(
            f"./data/{dataset}.data", sep = str(separators[i]), header = None))

    return data
