import os
import sys
import math
import george
import numpy as np
import pandas as pd
import logging
import json
import datetime

from utils import read_dataset, get_datasets_list, get_distance_between, read_full_result, read_opt_result
from objective_function import ML

logger = logging.getLogger(__name__)


def get_nearest_names(n, problem_name):
    df = pd.read_csv(os.path.join('../', 'datasets-distances.csv'))
    nearests = df.loc[df['dataset1'] == problem_name].sort_values(
        by=['distance'])['dataset2'].tolist()
    return nearests[1:(n + 1)]


def calc_local_dz(X_problem, y_problem, neighbor_name):
    dz = []
    dir = '../optimization_results/f-score/random-log_ei-gp/'
    X_neigbor, y_neighbor = read_opt_result(dir + neighbor_name + '/run-0')
    X_neigbor = np.array(X_neigbor)
    dz.append(abs(y_problem[0] - y_neighbor))

    return dz


def calc_reliability(problem_name, neighbor_name, X_problem, y_problem,
                     iteration):
    t = iteration
    alpha = 0.375
    d = get_distance_between(problem_name, neighbor_name)
    dz_local = calc_local_dz(X_problem, y_problem, neighbor_name)
    dz_global = np.average(dz_local)
    # here d and dz_global are already calculated kernels
    R = alpha**t - d + (1 - alpha**t) * dz_global
    logger.debug('Reliability for %s and %s on iteration %d: %f', problem_name,
                 neighbor_name, iteration, R)
    return R
