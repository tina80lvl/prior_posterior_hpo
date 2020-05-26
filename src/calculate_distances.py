import os
import sys
import math
import json
import datetime

from utils import read_dataset, get_datasets_list

import pandas as pd


def two_points(a, b):
    sum_sqr = 0
    for i, j in zip(a[1:], b[1:]):
        sum_sqr += (i - j)**2
    return math.sqrt(sum_sqr)


def calculate_distances():
    meta = pd.read_csv(
        os.path.join('../metrics/', 'normalized_meta_features.csv'))

    info = open('../datasets-distances.csv', 'w')
    info.write('"dataset1","dataset2","distance"\n')

    for dataset1 in meta.values:
        for dataset2 in meta.values:
            # print(dataset1[0], dataset2[0])
            distance = two_points(dataset1, dataset2)
            info.write(dataset1[0] + ',' + dataset2[0] + ',' + str(distance) +
                       '\n')


calculate_distances()
