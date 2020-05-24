import os
import sys
import math
import george
import numpy as np
import pandas as pd
import logging

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from robo.models.gaussian_process import GaussianProcess

logger = logging.getLogger(__name__)


class ML(object):
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def __call__(self, hypers):
        logger.debug('Hyperparameters configuration: ' + str(hypers))
        hidden_layer_sizes = int(hypers[0])
        alpha = float(hypers[1])
        learning_rate_init = float(hypers[2])
        max_iter = int(hypers[3])
        validation_fraction = float(hypers[4])
        beta_1 = float(hypers[5])
        beta_2 = float(hypers[6])
        n_iter_no_change = int(hypers[7])
        logger.debug('Hyperparameters configuration (tuned): [' +
                     str(hidden_layer_sizes) + ' ' + str(alpha) + ' ' +
                     str(learning_rate_init) + ' ' + str(max_iter) + ' ' +
                     str(validation_fraction) + ' ' + str(beta_1) + ' ' +
                     str(beta_2) + ' ' + str(n_iter_no_change) + ']')
        model = MLPClassifier(
            hidden_layer_sizes=(hidden_layer_sizes, ),
            activation='relu',
            solver='adam',
            alpha=alpha,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=learning_rate_init,
            power_t=0.5,  # only for 'sgb'
            max_iter=max_iter,
            shuffle=True,
            random_state=None,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            momentum=0.9,  # only for 'sgb'
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=1e-08,
            n_iter_no_change=n_iter_no_change)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)

        # calculating F-score
        f_score = f1_score(self.y_val, y_pred, average='macro')
        logger.debug('(1 - F-score): ' + str(f_score))

        return 1 - f_score
