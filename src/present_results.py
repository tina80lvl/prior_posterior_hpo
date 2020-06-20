import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import read_dataset, read_full_result, get_datasets_list, get_opt

DATASETS = [
    'abalone', 'artificial-characters', 'balance-scale', 'breast-tissue',
    'car', 'cardiotocography', 'cmc', 'cnae-9', 'collins', 'covertype',
    'desharnais', 'diggle_table_a2', 'ecoli', 'energy-efficiency',
    'eye_movements', 'fabert', 'fars', 'Fashion-MNIST', 'gas-drift',
    'gas-drift-different-concentrations', 'gina_prior2', 'glass', 'har',
    'hayes-roth', 'heart-long-beach', 'heart-switzerland', 'helena',
    'Indian_pines', 'iris', 'jannis', 'JapaneseVowels',
    'jungle_chess_2pcs_endgame_panther_elephant',
    'jungle_chess_2pcs_raw_endgame_complete', 'leaf',
    'LED-display-domain-7digit', 'mfeat-factors', 'mfeat-fourier',
    'mfeat-karhunen', 'mfeat-morphological', 'mfeat-pixel',
    'microaggregation2', 'nursery', 'page-blocks', 'pokerhand', 'PopularKids',
    'prnn_fglass', 'prnn_viruses', 'rmftsa_sleepdata', 'robot-failures-lp1',
    'robot-failures-lp5', 'satimage', 'seeds', 'segment', 'seismic-bumps',
    'semeion', 'shuttle', 'spectrometer', 'steel-plates-fault',
    'synthetic_control', 'tae', 'tamilnadu-electricity', 'teachingAssistant',
    'thyroid-allbp', 'thyroid-allhyper', 'user-knowledge', 'vehicle',
    'vertebra-column', 'volcanoes-a1', 'volcanoes-a3', 'volcanoes-a4',
    'volcanoes-d1', 'wall-robot-navigation', 'waveform-5000', 'wine',
    'wine-quality-white', 'zoo'
]


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


def present_incubment(dir_name, dataset_name, run, info_file):
    (x_opt, f_opt, incubments, incumbent_values, runtime, overhead, X,
     y) = read_full_result(dir_name + 'classical-bo/f-score/random-log_ei-gp/' + dataset_name + '/run-' + str(run))

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
             'cx',
             markersize=10,
             markeredgewidth=2,
             label='best: ' + str(ymin))
    plt.xlabel('Iteration')
    plt.ylabel('1 - F-score')
    plt.legend()
    plt.title(dataset_name)
    plt.savefig('../png/incubment-iteration/' + dataset_name + '.png')
    # plt.show()
    # plt.clf()


def present_mean_incubment(dir_name, dataset_name, info_file):
    all_incumbent_values = list()
    for run in range(10):
        (x_opt, f_opt, incubments, incumbent_values, runtime, overhead, X,
         y) = read_full_result(dir_name + 'classical-bo/f-score/random-log_ei-gp/' + dataset_name + '/run-' + str(run))
        all_incumbent_values.append(incumbent_values)

    mean_incumbent_values = list()
    for iter in range(len(all_incumbent_values[0])):
        sum = 0
        for run in range(len(all_incumbent_values)):
            sum += all_incumbent_values[run][iter]
        mean_incumbent_values.append(sum / 10)

    plt.plot(mean_incumbent_values,
             linestyle='solid',
             color='blue',
             label='incubment')

    ymin = min(mean_incumbent_values)
    xpos = mean_incumbent_values.index(ymin)
    xmin = xpos

    info_file.write(dataset_name + ',' + str(ymin) + ',' + str(xmin) + '\n')

    plt.plot(xmin,
             ymin,
             'cx',
             markersize=10,
             markeredgewidth=2,
             label='best: ' + str(ymin))
    plt.xlabel('Iteration')
    plt.ylabel('1 - F-score')
    plt.legend()
    plt.title(dataset_name)
    plt.savefig('../png/mean-incubment-iteration/' + dataset_name + '.png')
    # plt.show()
    # plt.clf()


def present_incubments(run):
    # datasets = get_datasets_list('../datasets/')

    info = open('../incubment-results.csv', 'w')
    info.write('"name","best_incubment","achieved_iteration"\n')

    for dataset in DATASETS:
        present_incubment('../optimization_results/f-score/random-log_ei-gp/',
                          dataset, run, info)
        plt.clf()


def present_mean_incubments():
    # datasets = get_datasets_list('../datasets/')

    info = open('../mean-incubment-results.csv', 'w')
    info.write('"name","best_incubment","achieved_iteration"\n')

    for dataset in DATASETS:
        present_mean_incubment(
            '../optimization_results/', dataset, info)
        plt.clf()


# ---------------------------------------------------------------------------- #


def present_incubment_posterior(dir_name, dataset_name, run1, run2, info_file):
    (x_opt1, f_opt1, incubments1, incumbent_values1, runtime1, overhead1, X1,
     y1) = read_full_result(dir_name +
                            'classical-bo/f-score/random-log_ei-gp/' +
                            dataset_name + '/run-' + str(run1))

    (x_opt2, f_opt2, incubments2, incumbent_values2, runtime2, overhead2, X2,
     y2) = read_full_result(dir_name +
                            'posterior-init/f-score/random-log_ei-gp/' +
                            dataset_name + '/run-' + str(run2))

    plt.plot(incumbent_values1,
             linestyle='solid',
             color='blue',
             label='incubment without posterior')
    plt.plot(incumbent_values2,
             linestyle='solid',
             color='magenta',
             label='incubment with posterior')

    ymin1 = min(incumbent_values1)
    xpos1 = incumbent_values1.index(ymin1)
    xmin1 = xpos1

    ymin2 = min(incumbent_values2)
    xpos2 = incumbent_values2.index(ymin2)
    xmin2 = xpos2

    # info_file.write(dataset_name + ',' + str(ymin1) + ',' + str(ymin2) + ',' +
    #                 str(xmin1) + ',' + str(xmin2) + '\n')

    info_file.write(dataset_name + ',' + str(ymin2) + ',' + str(xmin2) + '\n')

    plt.plot(xmin1,
             ymin1,
             'cx',
             markersize=10,
             markeredgewidth=2,
             label='best: ' + str(ymin1))
    plt.plot(xmin2,
             ymin2,
             'rx',
             markersize=10,
             markeredgewidth=2,
             label='best: ' + str(ymin2))
    plt.xlabel('Iteration')
    plt.ylabel('1 - F-score')
    plt.legend()
    plt.title(dataset_name)
    plt.savefig('../png/incubment-iteration-posterior/' + dataset_name +
                '.png')
    # plt.show()


def present_mean_incubment_posterior(dir_name, dataset_name, info_file):
    all_incumbent_values1 = list()
    all_incumbent_values2 = list()
    for run in range(10):
        (x_opt1, f_opt1, incubments1, incumbent_values1, runtime1, overhead1,
         X1, y1) = read_full_result(dir_name +
                                    'classical-bo/f-score/random-log_ei-gp/' +
                                    dataset_name + '/run-' + str(run))
        (x_opt2, f_opt2, incubments2, incumbent_values2, runtime2, overhead2,
         X2,
         y2) = read_full_result(dir_name +
                                'posterior-init/f-score/random-log_ei-gp/' +
                                dataset_name + '/run-' + str(run))

        all_incumbent_values1.append(incumbent_values1)
        all_incumbent_values2.append(incumbent_values2)

    mean_incumbent_values1 = list()
    mean_incumbent_values2 = list()
    for iter in range(len(all_incumbent_values1[0])):
        sum1 = 0
        sum2 = 0
        for run in range(len(all_incumbent_values1)):
            sum1 += all_incumbent_values1[run][iter]
            sum2 += all_incumbent_values2[run][iter]
        mean_incumbent_values1.append(sum1 / 10)
        mean_incumbent_values2.append(sum2 / 10)

    plt.plot(mean_incumbent_values1,
             linestyle='solid',
             color='blue',
             label='incubment without posterior')
    plt.plot(mean_incumbent_values2,
             linestyle='solid',
             color='magenta',
             label='incubment with posterior')

    ymin1 = min(mean_incumbent_values1)
    xpos1 = mean_incumbent_values1.index(ymin1)
    xmin1 = xpos1

    ymin2 = min(mean_incumbent_values2)
    xpos2 = mean_incumbent_values2.index(ymin2)
    xmin2 = xpos2

    # info_file.write(dataset_name + ',' + str(ymin1) + ',' + str(ymin2) + ',' +
    #                 str(xmin1) + ',' + str(xmin2) + '\n')

    info_file.write(dataset_name + ',' + str(ymin2) + ',' + str(xmin2) + '\n')

    plt.plot(xmin1,
             ymin1,
             'cx',
             markersize=10,
             markeredgewidth=2,
             label='best: ' + str(ymin1))
    plt.plot(xmin2,
             ymin2,
             'rx',
             markersize=10,
             markeredgewidth=2,
             label='best: ' + str(ymin2))
    plt.xlabel('Iteration')
    plt.ylabel('1 - F-score')
    plt.legend()
    plt.title(dataset_name)
    plt.savefig('../png/mean-incubment-iteration-posterior/' + dataset_name +
                '.png')
    # plt.show()


def present_incubments_posterior(run1=0, run2=0):
    info = open('../incubment-results-posterior.csv', 'w')
    info.write(
        '"name","best_incubment_posterior","achieved_iteration_posterior"\n'
    )

    for dataset in DATASETS:
        present_incubment_posterior('../optimization_results/', dataset, run1,
                                    run2, info)
        plt.clf()


def present_mean_incubments_posterior():
    info = open('../mean-incubment-results-posterior.csv', 'w')
    info.write(
        '"name","best_incubment_posterior","achieved_iteration_posterior"\n'
    )

    for dataset in DATASETS:
        present_mean_incubment_posterior('../optimization_results/', dataset,
                                         info)
        plt.clf()


# present_incubments(0)
present_mean_incubments()
# present_incubments_posterior(0, 0)
# present_incubments_posterior()
# present_mean_incubments_posterior()
