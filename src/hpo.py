import os
import sys
import math
import george
import numpy as np
import pandas as pd
import logging
import json
import datetime
import datetime

from utils import read_dataset
from utils import get_datasets_list
from objective_function import ML

from robo.priors.default_priors import DefaultPrior
# from robo.fmin.bayesian_optimization import bayesian_optimization
from my_bo import bayesian_optimization
from robo.priors.default_priors import DefaultPrior
from robo.solver.bayesian_optimization import BayesianOptimization
# from robo.models.random_forest import RandomForest
from robo.models.gaussian_process import GaussianProcess
from robo.acquisition_functions.log_ei import LogEI
from robo.initial_design import init_latin_hypercube_sampling
from robo.initial_design import init_random_uniform
from robo.maximizers.random_sampling import RandomSampling


def train_dataset(dataset_name, runs):
    data = read_dataset('../datasets/', dataset_name + '.csv')
    splitter = math.ceil(0.6 * len(data))

    train_data = data[:splitter]
    test_data = data[splitter:]

    # train
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data[train_data.columns.to_list()[-1]]
    # test
    X_val = test_data.iloc[:, :-1].values
    y_val = test_data[train_data.columns.to_list()[-1]]

    training_lower = np.min(X_train, axis=0)
    training_upper = np.max(X_train, axis=0)

    for i in range(runs):
        # initializing kernel with hyperparameters
        cov_amp = 2
        rng = np.random.RandomState(np.random.randint(0, 10000))
        n_dims = training_lower.shape[0]
        # print(n_dims)
        initial_ls = np.ones([n_dims])  # init hypers
        # print(initial_ls)
        exp_kernel = george.kernels.Matern52Kernel(initial_ls, ndim=n_dims)
        kernel = cov_amp * exp_kernel
        prior = DefaultPrior(len(kernel) + 1)

        # TODO make universalx for any model
        my_objective_function = ML(prior=prior,
                                   lower=training_lower,
                                   upper=training_upper,
                                   X_train=X_train,
                                   y_train=y_train,
                                   X_val=X_val,
                                   y_val=y_val,
                                   rng=rng)

        opt_lower = np.min(X_train, axis=0)  # size: number of hyperparameters
        opt_upper = np.max(X_train, axis=0)  # size: number of hyperparameters
        n_init = 3
        init_design = init_random_uniform
        n_iterations = 30
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

        results = bayesian_optimization(my_objective_function,
                                        opt_lower,
                                        opt_upper,
                                        num_iterations=n_iterations,
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


def train_datasets():
    logging.basicConfig(filename='../training_logs/' +
                        datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') +
                        '-training-debug.log',
                        level=logging.DEBUG)

    datasets = get_datasets_list('../datasets/')
    optimization_runs_per_dataset = 10
    for dataset_name in datasets:
        train_dataset(dataset_name, optimization_runs_per_dataset)


# train_datasets()
train_dataset('robot-failures-lp1', 1)
