import os
import sys
import math
import george
import numpy as np
import pandas as pd
import logging

from sklearn.metrics import f1_score
from robo.models.gaussian_process import GaussianProcess

logger = logging.getLogger(__name__)


class ML(object):
    def __init__(self, prior, lower, upper, X_train, y_train, X_val, y_val,
                 rng):
        self.prior = prior
        self.rng = rng
        self.lower = lower
        self.upper = upper
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def __call__(self, hypers):
        p
        cov_amp = 2
        n_dims = self.lower.shape[0]
        initial_ls = hypers  # init hypers
        exp_kernel = george.kernels.Matern52Kernel(initial_ls, ndim=n_dims)
        kernel = cov_amp * exp_kernel
        model = GaussianProcess(kernel,
                                prior=self.prior,
                                rng=self.rng,
                                normalize_output=False,
                                normalize_input=True,
                                lower=self.lower,
                                upper=self.upper)
        model.train(self.X_train, self.y_train)
        mean, _ = model.predict(self.X_val)
        y_pred = np.around(mean)

        # calculating F-score
        f_score = f1_score(self.y_val, y_pred, average='macro')
        logger.debug("(1 - F-score): " + str(f_score))

        return 1 - f_score
