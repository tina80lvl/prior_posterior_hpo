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
from define_neighbor import InitPosterior
from my_bo import bayesian_optimization
from objective_function import ML

from sklearn.model_selection import train_test_split

from robo.priors.default_priors import DefaultPrior
# from robo.fmin.bayesian_optimization import bayesian_optimization
from robo.priors.default_priors import DefaultPrior
from robo.solver.bayesian_optimization import BayesianOptimization
# from robo.models.random_forest import RandomForest
from robo.models.gaussian_process import GaussianProcess
from robo.acquisition_functions.log_ei import LogEI
from robo.initial_design import init_latin_hypercube_sampling
from robo.maximizers.random_sampling import RandomSampling

logging.basicConfig(filename='../training_logs/' +
                    datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') +
                    '-training-debug.log',
                    level=logging.DEBUG)


def train_dataset(dataset_name, initial_design, runs=1):
    data = read_dataset('../datasets/', dataset_name + '.csv')

    X_train, X_val, y_train, y_val = train_test_split(
        data.iloc[:, :-1].values, data[data.columns.to_list()[-1]])

    training_lower = np.min(X_train, axis=0)
    training_upper = np.max(X_train, axis=0)

    for i in range(runs):
        # TODO make universal for any model
        objective_function = ML(X_train=X_train,
                                y_train=y_train,
                                X_val=X_val,
                                y_val=y_val)

        opt_lower = np.array([1, 0.00001, 0.0001, 50, 0.01, 0.09, 0.0999,
                              5])  # size: number of hyperparameters
        opt_upper = np.array([150, 0.01, 0.1, 300, 0.9, 0.9, 0.999,
                              15])  # size: number of hyperparameters
        n_init = 3  # number of points for the initial design.
        init_design = init_latin_hypercube_sampling

        n_iterations = 100

        X_init = None  # mvp
        Y_init = None  # mvp

        maximizer = 'random'
        acquisition_func = 'log_ei'
        model_type = 'gp'
        result_path = ('../optimization_results/f-score/' + maximizer + '-' +
                       acquisition_func + '-' + model_type + '/' +
                       dataset_name + '/run-' + str(i))
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        results = bayesian_optimization(objective_function,
                                        opt_lower,
                                        opt_upper,
                                        num_iterations=n_iterations,
                                        initial_design=initial_design,
                                        X_init=X_init,
                                        Y_init=Y_init,
                                        maximizer=maximizer,
                                        acquisition_func=acquisition_func,
                                        model_type=model_type,
                                        n_init=3,
                                        rng=None,
                                        output_path=result_path)
        json.dump(results, open(os.path.join(result_path, 'RESULTS.json'),
                                'w'))


def train_datasets(initial_design, optimization_runs_per_dataset=1):
    datasets = get_datasets_list('../datasets/')
    for dataset_name in datasets:
        train_dataset(dataset_name, initial_design,
                      optimization_runs_per_dataset)


# without posterior
# train_datasets(init_latin_hypercube_sampling)

# with posterior
# train_dataset('name', InitPosterior())

train_dataset('PopularKids', init_latin_hypercube_sampling)
