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
from my_bo import bayesian_optimization
from objective_function import ML

logger = logging.getLogger(__name__)


class InitPosterior(object):
    def __init__(self, nearest_names):
        self.nearest_names = nearest_names

    def __call__(self, lower, upper, n_points, rng=None):
        dir = '../optimization_results/f-score/random-log_ei-gp/'
        samples = []
        for i in range(n_points):
            neighbor_name = self.nearest_names[i]
            X_nearest, y_nearest = read_opt_result(dir + neighbor_name)
            samples.append(X_nearest)
        return np.array(samples)
