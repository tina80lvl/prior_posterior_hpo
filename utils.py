import os
import sys
import pandas as pd

def read_dataset(dir_name, file_name):
    print('Measuring:  %s' % file_name) # TODO logging
    dataset = pd.read_csv(os.path.join(dir_name, file_name))
    # print(dataset)
    return dataset
