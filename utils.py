import os
import sys
from os import listdir
from os.path import isfile, join
import pandas as pd

def read_dataset(dir_name, file_name):
    print('Measuring:  %s' % file_name) # TODO logging
    dataset = pd.read_csv(os.path.join(dir_name, file_name))
    # print(dataset)
    return dataset

def get_datasets_list(path):
    return [f[:-4] for f in listdir(path) if isfile(join(path, f))]
