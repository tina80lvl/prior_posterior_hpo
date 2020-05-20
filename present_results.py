import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import read_dataset
from utils import read_result
from utils import get_datasets_list

def plot_predicted(mean, variance, real, dataset_name):
    plt.plot(mean, linestyle='dashed', color="red", label='mean')
    plt.plot(variance, linestyle='solid', color="blue", label='variance')
    plt.plot(real, linestyle='solid', color="green", label='real')
    plt.xlabel('Points to predict')
    plt.legend()
    plt.title(dataset_name)
    time = datetime.datetime.now().strftime('-%H-%M-%S')
    plt.savefig('png/' + dataset_name + time + '.png')
    plt.show()
    plt.clf()

def present_result(dir_name, dataset_name):
    (incumbents, x_opt, f_opt, incumbent_values, runtime, overhead, X, y,
        mean, variance, real) = read_result(dir_name + dataset_name)
    print('incubments', len(incumbents), len(incumbents[0]))
    print('incumbent_values', len(incumbent_values))

    print('runtime', len(runtime))
    print('overhead', len(overhead))

    print('X', len(X), len(X[0]))
    print('y', len(y))

    print('x_opt', len(x_opt))
    print('f_opt =', f_opt)

    print('mean', len(mean))
    print('variance', len(variance))
    print('real', len(real))

    plot_predicted(mean, variance, real, dataset_name)

present_result('optimization_results/differential_evolution-ei-gp/', 'page-blocks')
