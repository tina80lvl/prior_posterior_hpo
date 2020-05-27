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


def present_incubment(dir_name, dataset_name, info_file):
    (x_opt, f_opt, incubments, incumbent_values, runtime, overhead, X,
     y) = read_full_result(dir_name + dataset_name + '/run-0/')

    plt.plot(incumbent_values,
             linestyle='solid',
             color='blue',
             label='incubment')

    ymin = min(incumbent_values)
    xpos = incumbent_values.index(ymin)
    xmin = xpos

    info_file.write(dataset_name + ',' + str(ymin) + ',' + str(xmin) + '\n')

    plt.plot(xmin,
             ymin,
             'rx',
             markersize=10,
             markeredgewidth=2,
             label='best: ' + str(ymin))
    plt.xlabel('Iteration')
    plt.ylabel('1 - F-score')
    plt.legend()
    plt.title(dataset_name)
    plt.savefig('../png/incubment-iteration/' + dataset_name + '.png')
    # plt.show()
    plt.clf()


def present_all_incubments():
    # datasets = get_datasets_list('../datasets/')

    info = open('../incubment-results.csv', 'w')
    info.write('"name","best_incubment","achieved_iteration"\n')

    datasets = [
        'page-blocks', 'robot-failures-lp1', 'mfeat-fourier',
        'jungle_chess_2pcs_raw_endgame_complete', 'heart-switzerland',
        'gas-drift-different-concentrations', 'wall-robot-navigation',
        'jungle_chess_2pcs_endgame_panther_elephant', 'leaf', 'PopularKids',
        'mfeat-karhunen', 'diggle_table_a2', 'rmftsa_sleepdata', 'semeion',
        'desharnais', 'teachingAssistant', 'collins', 'volcanoes-a3',
        'artificial-characters', 'volcanoes-a4', 'glass', 'nursery', 'shuttle',
        'segment', 'heart-long-beach', 'vertebra-column', 'cnae-9', 'jannis',
        'wine-quality-white', 'vehicle', 'ecoli', 'eye_movements', 'seeds',
        'car', 'fabert', 'breast-tissue', 'thyroid-allbp', 'gas-drift',
        'mfeat-factors', 'volcanoes-d1', 'har', 'satimage', 'Fashion-MNIST',
        'seismic-bumps', 'pokerhand', 'helena', 'thyroid-allhyper', 'wine',
        'balance-scale', 'microaggregation2', 'steel-plates-fault', 'tae',
        'mfeat-pixel', 'gina_prior2', 'synthetic_control', 'cmc',
        'energy-efficiency', 'iris', 'fars', 'abalone', 'prnn_viruses',
        'Indian_pines', 'covertype', 'JapaneseVowels', 'user-knowledge',
        'spectrometer', 'hayes-roth', 'robot-failures-lp5', 'prnn_fglass',
        'waveform-5000', 'zoo', 'cardiotocography', 'mfeat-morphological',
        'volcanoes-a1', 'tamilnadu-electricity', 'LED-display-domain-7digit'
    ]
    for dataset in datasets:
        present_incubment('../optimization_results/f-score/random-log_ei-gp/',
                          dataset, info)


present_all_incubments()
# present_incubment('../optimization_results/f-score/random-log_ei-gp/', 'PopularKids')
