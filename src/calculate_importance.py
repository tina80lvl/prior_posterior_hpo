import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import read_dataset, read_full_result, get_datasets_list, get_opt

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
    for name in datasets:
        mean_iter_disp, mean_val_disp = dispersion_per_dataset(name, '../optimization_results/f-score/random-log_ei-gp/')
        print(name, mean_iter_disp, mean_val_disp)

calculate_dispersions()
