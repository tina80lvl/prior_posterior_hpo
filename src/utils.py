import os
import sys
import json
from os import listdir
from os.path import isfile, join
import pandas as pd

def read_dataset(dir_name, file_name):
    print('Measuring:  %s' % file_name) # TODO logging
    dataset = pd.read_csv(os.path.join(dir_name, file_name))
    # print(dataset)
    return dataset

def get_results_fields(data):
    incumbents = data['incumbents']
    x_opt = data['x_opt']
    f_opt = data['f_opt']
    incumbent_values = data['incumbent_values']
    runtime = data['runtime']
    overhead = data['overhead']
    X = data['X']
    y = data['y']

    mean = data['mean']
    variance = data['variance']
    real = data['real']
    return (incumbents, x_opt, f_opt, incumbent_values, runtime,
        overhead, X, y, mean, variance, real)

def save_model(name, model):
    print('Saving model:  %s' % name)
    f = open('saved_models/' + name + '_model.txt', 'w')
    f.write(model)
    f.close()

def read_result(dir_name):
    with open(dir_name + '/RESULTS.json', 'r') as read_file:
        data = json.load(read_file)
        return get_results_fields(data)

def get_datasets_list(path):
    return [f[:-4] for f in listdir(path) if isfile(join(path, f))]
