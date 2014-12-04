#!/usr/bin/python
# -*- coding: utf-8 -*-

r"""
==========================================================
Comparing Bayesian Inference and MAP estimation
==========================================================

"""
print(__doc__)

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Licence: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt

from sklearn.learning_curve import learning_curve
from skgp.estimators import GaussianProcess, BayesianGaussianProcess

np.random.seed(1)


def f(X):
    """ Target function for GPR.

    Note that one dimension (X[:, 3]) is irrelevant and should thus be ignored
    by ARD. Furthermore, the values x in R^3 and
    x + \alpha (1, 2 , 0) + \beta (1, 0, 2) have the same value for all x and
    all alpha and beta. This can be exploited by FAD.
    """
    return np.tanh(2 * X[:, 0] - X[:, 1] - X[:, 2])

Xtrain = np.random.random((100, 6)) * 2 - 1
ytrain = f(Xtrain)

plt.figure()
colors = ['r', 'g', 'b', 'c', 'm']
labels = {True: "Bayesian GP", False: "Standard GP"}
for i, bayesian in enumerate(labels.keys()):
    model = GaussianProcess(corr='squared_exponential',
                            theta0=[1.0] * 12, thetaL=[1e-4] * 12,
                            thetaU=[1e2] * 12)
    if bayesian:
        model = BayesianGaussianProcess(model, n_posterior_samples=25,
                                        n_burnin=250, n_sampling_steps=25)

    train_sizes, train_scores, test_scores = \
        learning_curve(model, Xtrain, ytrain, scoring="mean_squared_error",
                       cv=10, n_jobs=1)
    test_scores = -test_scores  # Scores correspond to negative MSE
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_min = np.min(test_scores, axis=1)
    test_scores_max = np.max(test_scores, axis=1)

    plt.plot(train_sizes, test_scores_mean, label=labels[bayesian],
             color=colors[i])
    plt.fill_between(train_sizes, test_scores_min, test_scores_max,
                     alpha=0.2, color=colors[i])

plt.legend(loc="best")
plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Mean Squared Error")
plt.yscale("symlog", linthreshy=1e-10)
plt.show()
