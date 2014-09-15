.. -*- mode: rst -*-

skgp
====

skgp is an extension package for the sklearn (scikit-learn) package which extends sklearn's functionality in the area of Gaussian Processes (GPs). It builds heavily on sklearn's original GP implementation and some of skgp's functionality may be merged back into sklearn at some point in the future. skgp is distributed under the 3-Clause BSD license.

Features and Changes
====================
skgp's main features/changes compared to the standard GPs from sklearn are the following:

CHANGES:

  *  Correlation models are now classes instead of functions and inherit from abstract base class StationaryCorrelation. This reduces code-redundancy (DRY) and separates GPs and their correlation models more strictly. Moreover, non-stationary correlation models become possible (see examples/plot_gp_nonstationary.py) This was not possible formerly since only the differences of the datapoints were passed to the correlation models but not the datapoints themselves.
  *  The hyperparameters theta are estimated as maximum-a-posterior estimates rather than maximum-likelihood estimates. At the moment, the prior on the hyperparameters theta is uniform such that both result in the same value. However, a non-uniform prior could be used in the future in cases where the risk of overfitting theta to the data is serious (e.g., forcing the factor analysis distance to be close to diagonal etc.)

NEW FEATURES:

  *  Matern correlation models for nu=1.5 and nu=2.5 have been added (see https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function). An example script showing the potential benefit of the Matern correlation model compared to squared-exponential and absolute-exponential was added under examples/gaussian_process/plot_matern_kernel.py (see attached image)
  *  squared_exponential, absolute_exponential and Matern correlation models support factor analysis distance. This can be seen as an extension of learning dimension-specific length scales in which also correlations of feature dimensions can be taken into account. See Rasmussen and Williams 2006, p107 for details. An example script showing the potential benefit of this extension was added under examples/gaussian_process/plot_gp_learning_curve.py (see attached image).
  *  GP supports additional optimizers for ML estimation passed as callables. In addition to 'fmin_cobyla' and 'Welch', GaussianProcess allows now to pass other optimizers directly as callables via the parameter optimizer.
  * BayesianGaussianProcess which performs Bayesian inference of the hyperparameters using MCMC sampling based on the emcee package rather than employing ML or MAP point estimates.
  * Non-stationary correlation models where the non-stationarity is learned from the data.



Important links
===============

- Official source code repo: https://github.com/jmetzen/skgp
- sklearn: http://scikit-learn.org


Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --user

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install



