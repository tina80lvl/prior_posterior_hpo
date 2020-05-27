import os
import sys
import json
from os import listdir
from os.path import isfile, join
import pandas as pd


def read_dataset(dir_name, file_name):
    print('Measuring:  %s' % file_name)  # TODO logging
    dataset = pd.read_csv(os.path.join(dir_name, file_name))
    # print(dataset)
    return dataset


def get_distance_between(dataset_name1, dataset_name2):
    df = pd.read_csv(os.path.join('../', 'datasets-distances.csv'))
    return float(df.loc[df['dataset1'] == dataset_name1].loc[
        df['dataset2'] == dataset_name2]['distance'])


def get_results_fields(data):
    incumbents = data['incumbents']
    x_opt = data['x_opt']
    f_opt = data['f_opt']
    incumbent_values = data['incumbent_values']
    runtime = data['runtime']
    overhead = data['overhead']
    X = data['X']
    y = data['y']

    return x_opt, f_opt, incumbents, incumbent_values, runtime, overhead, X, y


def get_opt(data):
    x_opt, f_opt, _, _, _, _, _, _ = get_results_fields(data)
    return x_opt, f_opt


def read_full_result(dir_name):
    with open(dir_name + '/run-0/RESULTS.json', 'r') as read_file:
        data = json.load(read_file)
        return get_results_fields(data)


def read_opt_result(dir_name):
    with open(dir_name + '/run-0/RESULTS.json', 'r') as read_file:
        data = json.load(read_file)
        return get_opt(data)


def save_model(name, model):
    print('Saving model:  %s' % name)
    f = open('saved_models/' + name + '_model.txt', 'w')
    f.write(model)
    f.close()


def get_datasets_list(path):
    return [f[:-4] for f in listdir(path) if isfile(join(path, f))]
