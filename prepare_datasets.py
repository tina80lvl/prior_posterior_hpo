import pandas as pd
from utils import read_dataset

datasets = ['small']
for dataset_name in datasets:
    data = read_dataset('datasets/', dataset_name + '.csv')

    column = data.columns[-1]
    data[column] = pd.factorize(data[column])[0] + 1

    shuffle = data.sample(frac=1) # shuffle rows
    shuffle.to_csv('datasets/'+ dataset_name + '.csv', index=False)
