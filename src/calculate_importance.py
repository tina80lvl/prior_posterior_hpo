import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from utils import read_dataset, read_full_result, read_opt_result, get_datasets_list, get_opt

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


def dispersion_per_dataset(dataset_name, dir_name):
    iterations = list()
    values = list()
    for run in range(10):
        (x_opt, f_opt, incubments, incumbent_values, runtime, overhead, X,
         y) = read_full_result(dir_name + dataset_name + '/run-' + str(run))
        ymin = f_opt
        xpos = incumbent_values.index(ymin)
        xmin = xpos
        iterations.append(xmin)
        values.append(ymin)

    mean_iter = sum(iterations) / len(iterations)
    mean_val = sum(values) / len(values)

    iteration_disp = list()
    values_disp = list()
    for i in range(10):
        it = iterations[i]
        val = values[i]
        iteration_disp.append(abs(it - mean_iter))
        values_disp.append(abs(val - mean_val))

    mean_iter_disp = sum(iteration_disp) / len(iteration_disp)
    mean_val_disp = sum(values_disp) / len(values_disp)

    return round(mean_iter_disp), mean_val_disp


def calculate_dispersions():
    # c_dir = '../optimization_results/classical-bo/f-score/random-log_ei-gp/'
    p_dir = '../optimization_results/posterior-init/f-score/random-log_ei-gp/'
    info = open('../dispersion-pbo.csv', 'w')
    info.write(
        '"name","mean disp (val)","mean disp (iter)"\n'
    )
    for name in DATASETS:
        mean_iter_disp, mean_val_disp = dispersion_per_dataset(
            name, p_dir)
        info.write(name + ',' + str(mean_val_disp) + ',' + str(mean_iter_disp) + '\n')
        print(name, mean_val_disp, mean_iter_disp)


def calculate_importance(dataset_name):
    c_dir = '../optimization_results/classical-bo/f-score/random-log_ei-gp/'
    p_dir = '../optimization_results/posterior-init/f-score/random-log_ei-gp/'
    cbo_result = list()
    pbo_result = list()
    cbo_iters = list()
    pbo_iters = list()
    for run in range(10):
        c_x_opt, c_f_opt, c_incubments, c_incumbent_values, _, _, _, _ = read_full_result(
            c_dir + dataset_name + '/run-' + str(run))
        c_ymin = c_f_opt
        c_xpos = c_incumbent_values.index(c_ymin)
        c_xmin = c_xpos
        cbo_iters.append(c_xmin)
        cbo_result.append(c_ymin)

        p_x_opt, p_f_opt, p_incubments, p_incumbent_values, _, _, _, _ = read_full_result(
            p_dir + dataset_name + '/run-' + str(run))
        p_ymin = p_f_opt
        p_xpos = p_incumbent_values.index(p_ymin)
        p_xmin = p_xpos
        pbo_iters.append(p_xmin)
        pbo_result.append(p_ymin)

        # _, classical_f_opt = read_opt_result(
        #     '../optimization_results/classical-bo/f-score/random-log_ei-gp/' +
        #     dataset_name + '/run-' + str(run))
        # _, posterior_f_opt = read_opt_result(
        #     '../optimization_results/posterior-init/f-score/random-log_ei-gp/'
        #     + dataset_name + '/run-' + str(run))
        # cbo_result.append(classical_f_opt)
        # pbo_result.append(posterior_f_opt)
        # print(classical_f_opt > posterior_f_opt)

    # print(cbo_result)
    # print(pbo_result)
    try:
        stat_res, pval_res = mannwhitneyu(cbo_result, pbo_result)
    except ValueError:
        stat_res = pval_res = -1
    try:
        stat_iter, pval_iter = mannwhitneyu(cbo_iters, pbo_iters)
    except ValueError:
        stat_iter, pval_iter  = -1

    # print(stat, pval)
    return stat_res, pval_res, stat_iter, pval_iter


def importance():
    info = open('../importance.csv', 'w')
    info.write(
        '"name","statistic (val)","pvalue (val)","statistic (iter)","pvalue (iter)"\n'
    )
    for name in DATASETS:
        stat_res, pval_res, stat_iter, pval_iter = calculate_importance(
            name)
        info.write(name + ',' + str(stat_res) + ',' + str(pval_res) + ',' +
                   str(stat_iter) + ',' + str(pval_iter) + '\n')
        # print(name, stat, pval)


# importance()
# calculate_importance('wine')
calculate_dispersions()
