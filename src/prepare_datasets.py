import pandas as pd
from utils import read_dataset
from utils import get_datasets_list


def prepare_dataset(dataset_name):
    data = read_dataset('../datasets/', dataset_name + '.csv')

    for i in range(len(data.columns) - 1):
        column = data.columns[i]
        data[column] = pd.factorize(data[column])[0] + 1

    # shuffle = data.sample(frac=1)  # shuffle rows
    # if len(shuffle) > 5000:
    #     shuffle = shuffle[:5000]  # cut objects
    # shuffle.to_csv('../datasets/' + dataset_name + '.csv', index=False)
    data.to_csv('../datasets/' + dataset_name + '.csv', index=False)


def prepare_all():
    datasets = get_datasets_list('../datasets/')
    for dataset_name in datasets:
        prepare_dataset(dataset_name)


# prepare_dataset('collins')
prepare_dataset('zoo')
# prepare_all()
