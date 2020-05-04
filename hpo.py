import os
import sys
import numpy as np
import pandas as pd
import logging
# import matplotlib.pyplot as plt
import datetime

from robo.models.bayesian_linear_regression import BayesianLinearRegression

def read_dataset(dir_name, file_name):
    print('Measuring:  %s' % file_name)
    dataset = pd.read_csv(os.path.join(dir_name, file_name))
    # print(dataset)
    return dataset

def objective_function(x):
    y = np.sin(3 * x[0]) * 4 * (x[0] - 1) * (x[0] + 2)
    return y

def train_model(dataset):
    X = dataset.iloc[:, :-1].values
    y = dataset[dataset.columns.to_list()[-1]]
    model = BayesianLinearRegression()
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

# def plot_predicted(mean, variance, real, name):
#     plt.plot(mean, linestyle='dashed', color="red", label='mean')
#     plt.plot(variance, linestyle='solid', color="blue", label='variance')
#     plt.plot(real, linestyle='solid', color="green", label='real')
#     plt.xlabel('Points to predict')
#     plt.legend()
#     time = datetime.datetime.now().strftime('-%H-%M-%S')
#     plt.savefig('png/' + name + time + '.png')
#     plt.clf()

def main():
    logging.basicConfig(level=logging.INFO)

    # datasets = ['small']
    datasets = ['olivetti', 'umist', 'poker', 'eating', 'mouse', 'fashion']
    for dataset_name in datasets:
        train_data = read_dataset('datasets/', dataset_name + '_train.csv')
        model = train_model(train_data)

        test_data = read_dataset('datasets/', dataset_name + '_test.csv')
        mean, var, real = predict(model, test_data)

        # plot_predicted(mean, var, real, dataset_name)

    # lower = np.array([0])
    # upper = np.array([6])
    #
    # results = bayesian_optimization(objective_function, lower, upper, num_iterations=50)

main()
