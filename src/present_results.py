import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import read_dataset, read_full_result, get_datasets_list, get_opt


def plot_predicted(mean, variance, real, dataset_name):
    plt.plot(mean, linestyle='dashed', color="red", label='mean')
    plt.plot(variance, linestyle='solid', color="blue", label='variance')
    plt.plot(real, linestyle='solid', color="green", label='real')
    plt.xlabel('Points to predict')
    plt.legend()
    plt.title(dataset_name)
    time = datetime.datetime.now().strftime('-%H-%M-%S')
    plt.savefig('../png/' + dataset_name + time + '.png')
    plt.show()
    plt.clf()


def present_incubment(dir_name, dataset_name):
    (incumbents, x_opt, f_opt, incumbent_values, runtime, overhead, X, y) = read_full_result(dir_name + dataset_name + '/run-0/')

    plt.plot(incumbent_values, linestyle='solid', color='blue', label='incubment')
    plt.xlabel('Iteration')
    plt.ylabel('1 - F-score')
    plt.legend()
    plt.title(dataset_name)
    # time = datetime.datetime.now().strftime('-%H-%M-%S')
    plt.savefig('../png/incubment-iteration/' + dataset_name + '.png')
    plt.show()
    plt.clf()


def present_all_incubments():
    datasets = get_datasets_list('../datasets/')
    for dataset in datasets:
        present_incubment('../optimization_results/f-score/random-log_ei-gp/', dataset)


# present_all_incubments()
present_incubment('../optimization_results/f-score/random-log_ei-gp/', 'PopularKids')
# present_result('optimization_results/differential_evolution-ei-gp/',
#                'page-blocks')
