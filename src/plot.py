import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import datetime


def main():
    file_name = 'presentation'
    n_datasets = 1
    with open('../results/' + file_name + '.txt') as fp:
        for i in range(n_datasets):
            dataset_name = fp.readline()[12:-5]
            fp.readline()
            fp.readline()
            mean = list(map(float, fp.readline()[1:-2].split(',')))
            fp.readline()
            variance = list(map(float, fp.readline()[1:-2].split(',')))
            # cheat_var = list()
            # for el in variance:
            #     cheat_var.append(el * 99)
            fp.readline()
            real = list(map(float, fp.readline()[1:-2].split(',')))
            # print(dataset_name, type(real), real)
            plot_predicted(mean, variance, real, dataset_name, file_name)

def plot_predicted(mean, variance, real, dataset_name, file_name):
    # plt.plot(mean, linestyle='dashed', color="red", label='mean')
    plt.plot(variance, linestyle='solid', color="blue", label='variance')
    # plt.plot(real, linestyle='solid', color="green", label='real')
    plt.xlabel('Points to predict')
    plt.legend()
    plt.title(dataset_name)
    time = datetime.datetime.now().strftime('-%H-%M-%S')
    plt.savefig('../png/' + file_name + '-' + dataset_name + time + '.png')
    plt.show()
    plt.clf()


main()
