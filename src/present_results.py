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


def present_incubment(dir_name, dataset_name, run, info_file):
    (x_opt, f_opt, incubments, incumbent_values, runtime, overhead, X,
     y) = read_full_result(dir_name + dataset_name + '/run-' + str(run))

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
         y) = read_full_result(dir_name + dataset_name + '/run-' + str(run))
        all_incumbent_values.append(incumbent_values)

    mean_incumbent_values = list()
    for iter in range(len(all_incumbent_values[0])):
        sum = 0
        for run in range(len(all_incumbent_values)):
            sum += all_incumbent_values[run][iter]
        mean_incumbent_values.append(sum/10)

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

def present_all_incubments(run):
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
                          dataset, run, info)
        plt.clf()

def present_all_mean_incubments(run):
    # datasets = get_datasets_list('../datasets/')

    info = open('../mean-incubment-results.csv', 'w')
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
        present_mean_incubment('../optimization_results/f-score/random-log_ei-gp/',
                          dataset, info)
        plt.clf()


def present_incubment_posterior(dir_name, dataset_name, run1, run2, info_file):
    (x_opt1, f_opt1, incubments1, incumbent_values1, runtime1, overhead1, X1,
     y1) = read_full_result(dir_name + 'f-score/random-log_ei-gp/' +
                            dataset_name + '/run-' + str(run1))

    (x_opt2, f_opt2, incubments2, incumbent_values2, runtime2, overhead2, X2,
     y2) = read_full_result(dir_name + 'posterior-init/' + dataset_name +
                            '/f-score/random-log_ei-gp/' + dataset_name +
                            '/run-' + str(run2))

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

    info_file.write(dataset_name + ',' + str(ymin1) + ',' + str(ymin2) + ',' +
                    str(xmin1) + ',' + str(xmin2) + '\n')

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
        (x_opt1, f_opt1, incubments1, incumbent_values1, runtime1, overhead1, X1,
         y1) = read_full_result(dir_name + 'f-score/random-log_ei-gp/' +
                                dataset_name + '/run-' + str(run))
        (x_opt2, f_opt2, incubments2, incumbent_values2, runtime2, overhead2, X2,
         y2) = read_full_result(dir_name + 'posterior-init/' + dataset_name +
                                '/f-score/random-log_ei-gp/' + dataset_name +
                                '/run-' + str(run))

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
        mean_incumbent_values1.append(sum1/10)
        mean_incumbent_values2.append(sum2/10)

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

    info_file.write(dataset_name + ',' + str(ymin1) + ',' + str(ymin2) + ',' +
                    str(xmin1) + ',' + str(xmin2) + '\n')

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


def present_incubments_with_posterior(run1=0, run2=0):
    info = open('../incubment-results-posterior.csv', 'w')
    info.write(
        '"name","best_incubment","best_incubment_posterior","achieved_iteration","achieved_iteration_posterior"\n'
    )

    datasets = [
        'car', 'wine', 'cmc', 'zoo', 'nursery', 'abalone', 'cardiotocography',
        'desharnais', 'glass', 'segment'
    ]
    for dataset in datasets:
        present_incubment_posterior('../optimization_results/', dataset, run1,
                                    run2, info)
        plt.clf()


def present_mean_incubments_with_posterior():
    info = open('../mean-incubment-results-posterior.csv', 'w')
    info.write(
        '"name","best_incubment","best_incubment_posterior","achieved_iteration","achieved_iteration_posterior"\n'
    )

    datasets = [
        'car', 'wine', 'cmc', 'zoo', 'nursery', 'abalone', 'cardiotocography',
        'desharnais', 'glass', 'segment'
    ]
    for dataset in datasets:
        present_mean_incubment_posterior('../optimization_results/', dataset, info)
        plt.clf()

present_all_incubments(0)

present_all_mean_incubments(0)
# present_incubments_with_posterior(0, 0)
present_incubments_with_posterior()
present_mean_incubments_with_posterior()
# present_incubment('../optimization_results/f-score/random-log_ei-gp/', 'PopularKids')
