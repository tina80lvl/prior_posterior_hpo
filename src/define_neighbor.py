import os
import sys
import math
import george
import numpy as np
import pandas as pd
import logging
import json
import datetime

from utils import read_dataset, get_datasets_list, get_distance_between
from my_bo import bayesian_optimization
from objective_function import ML

logger = logging.getLogger(__name__)


class InitPosterior(object):
    def __init__(self, X_nearest, y_nearest):
        # self.n_nearest = X_nearest.shape[0]
        self.X_nearest = X_nearest
        self.y_nearest = y_nearest

    def __call__(self, lower, upper, n_points, rng=None):
        n_dims = lower.shape[0]
        # Generate bounds for random number generator
        s_bounds = np.array([
            np.linspace(lower[i], upper[i], n_points + 1)
            for i in range(n_dims)
        ])
        s_lower = s_bounds[:, :-1]
        s_upper = s_bounds[:, 1:]
        print(s_bounds)
        print(s_lower)
        print(s_upper)


def get_nearest_names(n, problem_name):
    df = pd.read_csv(os.path.join('../', 'datasets-distances.csv'))
    nearests = df.loc[df['dataset1'] == problem_name].sort_values(
        by=['distance'])['dataset2'].tolist()
    return nearests[1:(n + 1)]


def calc_local_dz(X, y_problem, y_neighbor):
    dz = []
    for i in range(y_problem.shape[0]):
        dz.append(abs(y_problem[i] - y_neighbor[i]))
    return dz


def calc_reliability(problem_name, neighbor_name, X, y_problem, y_neighbor,
                     iteration):
    t = iteration
    alpha = 0.375
    d = get_distance_between(problem_name, neighbor_name)
    dz_local = calc_local_dz(X, y_problem, y_neighbor)
    dz_global = np.average(dz_local)
    # here d and dz_global are already calculated kernels
    R = alpha**t - d + (1 - alpha**t) * dz_global
    logger.debug('Reliability for %s and %s on iteration %d: %f', problem_name,
                 neighbor_name, iteration, R)
    return R
