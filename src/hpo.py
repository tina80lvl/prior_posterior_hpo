import os
import sys
import math
import george
import numpy as np
import pandas as pd
import logging
import json
import datetime
# import matplotlib.pyplot as plt
import datetime

from utils import read_dataset
from utils import get_datasets_list

from sklearn.metrics import f1_score

from robo.fmin.bayesian_optimization import bayesian_optimization
from robo.priors.default_priors import DefaultPrior
from robo.solver.bayesian_optimization import BayesianOptimization
# from robo.models.random_forest import RandomForest
from robo.models.gaussian_process import GaussianProcess
from robo.acquisition_functions.log_ei import LogEI
from robo.initial_design import init_latin_hypercube_sampling
from robo.initial_design import init_random_uniform
from robo.maximizers.random_sampling import RandomSampling

def save_model(name, model):
    print('Saving model:  %s' % name)
    f = open('saved_models/' + name + '_model.txt', 'w')
    f.write(model)
    f.close()

def train_model(dataset):
    X = dataset.iloc[:, :-1].values
    y = dataset[dataset.columns.to_list()[-1]]
    model = RandomForest()
    model.train(X, y)
    return model

def predict(model, test_data):
    x = test_data.iloc[:, :-1].values
    y = test_data[test_data.columns.to_list()[-1]]

    # print('predict')
    # print(x)
    mean_pred, var_pred = model.predict(x)
    print('-' * 15 + 'mean' + '-' * 15)
    print(list(mean_pred))
    print('-' * 15 + 'variance' + '-' * 15)
    print(list(var_pred))
    print('-' * 15 + 'real' + '-' * 15)
    np.set_printoptions(threshold=sys.maxsize)
    print(list(y))

    return mean_pred, var_pred, y

def objective_function(x):
    print('objective_func:', x)
    y = np.sin(3 * x[0]) * 4 * (x[0] - 1) * (x[0] + 2)
    return y

def main():
    logging.basicConfig(filename='../training_logs/' + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+'-training.log', level=logging.INFO)

    # datasets = get_datasets_list('datasets/')
    datasets = ['small']
    for dataset_name in datasets:
        data = read_dataset('datasets/', dataset_name + '.csv')
        splitter = math.ceil(0.6 * len(data))

        train_data = data[:splitter]
        test_data = data[splitter:]

        # train
        X = train_data.iloc[:, :-1].values
        y = train_data[train_data.columns.to_list()[-1]]
        # test
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data[train_data.columns.to_list()[-1]]

        lower = np.min(X, axis=0)
        upper = np.max(X, axis=0)
        n_init = 3
        init_design = init_random_uniform
        n_iterations = 5
        X_init = None # mvp
        Y_init = None # mvp

        maximizer = 'random'
        acquisition_func = 'log_ei'
        model_type = 'gp'
        result_path = ('optimization_results/f-score/' + maximizer + '-' +
            acquisition_func + '-' + model_type + '/' + dataset_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # my_model = GaussianProcess(....)
        # my_model.train()

        results = bayesian_optimization(objective_function, lower, upper,
            num_iterations=n_iterations, X_init=X_init, Y_init=Y_init,
            maximizer=maximizer, acquisition_func=acquisition_func,
            model_type=model_type, n_init=3, rng=None, output_path=result_path)
        json.dump(results, open(os.path.join(result_path, 'RESULTS.json'), 'w'))


main()