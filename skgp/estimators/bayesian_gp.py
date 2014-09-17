# -*- coding: utf-8 -*-

# Author: Jan Hendrik Metzen <jhmqinformatik.uni-bremen.de>
# Licence: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, clone


class BayesianGaussianProcess(BaseEstimator, RegressorMixin):
    """The Bayesian Gaussian Process model class.

    This class wraps a Gaussian Process object and allows to use sampling
    the hyperparameter from the posterior rather than selecting them using
    maximum likelihood. It thus accounts better for the uncertainty in
    hyperparameter selection (when computing the predictive variance) but at
    the cost of considerably improved computation time - both during fitting,
    which involves running MCMC with the emcee package, and during prediction,
    which requires to evaluate the base GP for every hyperparameter sample
    of the posterior.

    Parameters
    ----------
    base_gp : instance of Gaussian Process
        The basic Gaussian process which allows to train and execute a GP for
        fixed hyperparameters, which can be selected using maximum likelihood

    n_posterior_samples: int
        The number of samples taken from the posterior distribution. The more
        samples, the more accurate the posterior distribution is captured;
        however, the computational cost during prediction also increases
        linearly

    n_burnin: int
        The number of burn-in steps during MCMC sampling

    n_sampling_steps: int
        The number of actual sampling steps during MCMC sampling
    """

    def __init__(self, base_gp, n_posterior_samples=25,
                 n_burnin=500, n_sampling_steps=250):
        self.base_gp = base_gp
        self.n_posterior_samples = n_posterior_samples
        self.n_burnin = n_burnin
        self.n_sampling_steps = n_sampling_steps

        self.gps = []

    def fit(self, X, y):
        """
        The Bayesian Gaussian Process model fitting method.

        Parameters
        ----------
        X : double array_like
            An array with shape (n_samples, n_features) with the input at which
            observations were made.

        y : double array_like
            An array with shape (n_samples, ) or shape (n_samples, n_targets)
            with the observations of the output to be predicted.

        Returns
        -------
        gp : self
            A fitted Bayesian Gaussian Process model object awaiting data to
            perform predictions.
        """
        try:
            import emcee
        except ImportError:
            raise Exception("BayesianGaussianProcess requires the emcee "
                            "package")

        # Initialize with ML estimate
        self.base_gp.fit(X, y)

        # Initialize the MCMC sampling
        n_dim = len(self.base_gp.theta_)
        self.sampler = emcee.EnsembleSampler(nwalkers=2*n_dim, dim=n_dim,
                                             lnpostfn=self.lnpostfn)

        # Start ensemble walkers from perturbed ML estimate
        p0 = [self.base_gp.theta_ * (1 + np.random.randn(n_dim) * 2e-3 - 1e-3)
              for i in range(2*n_dim)]

        # Run burn-in
        p0, _, _ = self.sampler.run_mcmc(p0, self.n_burnin)

        # Running production chain
        self.sampler.run_mcmc(p0, self.n_sampling_steps)

        # Select posterior samples and create clone of GP for these
        self.theta_ = []
        for i in range(self.n_posterior_samples):
            # Select posterior samples
            w = np.random.randint(self.sampler.chain.shape[0])
            n = np.random.randint(self.n_burnin, self.sampler.chain.shape[1])
            self.theta_.append(self.sampler.chain[w, n])
            # Create clone of GP with these hyperparameter sample
            gp = clone(self.base_gp)
            gp.theta0 = self.theta_[-1]
            gp.thetaL = None
            gp.thetaU = None
            gp.fit(X, y)
            self.gps.append(gp)
        self.theta_ = np.array(self.theta_)

        # Make some of the base_gp's attributes accessible
        self.X = self.base_gp.X
        self.y = self.base_gp.y
        self.corr = self.base_gp.corr

        return self

    def predict(self, X, eval_MSE=False):
        """ Evaluates the Bayesian Gaussian Process model at X.

        Parameters
        ----------
        X : array_like
            An array with shape (n_eval, n_features) giving the point(s) at
            which the prediction(s) should be made.

        eval_MSE : boolean, optional
            A boolean specifying whether the Mean Squared Error should be
            evaluated or not.
            Default assumes evalMSE = False and evaluates only the BLUP (mean
            prediction).

        Returns
        -------
        y : array_like, shape (n_samples, ) or (n_samples, n_targets)
            An array with shape (n_eval, ) if the Gaussian Process was trained
            on an array of shape (n_samples, ) or an array with shape
            (n_eval, n_targets) if the Gaussian Process was trained on an array
            of shape (n_samples, n_targets) with the Best Linear Unbiased
            Prediction at x.

        MSE : array_like, optional (if eval_MSE == True)
            An array with shape (n_eval, ) or (n_eval, n_targets) as with y,
            with the Mean Squared Error at x.
        """
        X = np.atleast_2d(X)
        if self.y.shape[1] > 1:
            y_pred_means = \
                np.empty((self.n_posterior_samples, X.shape[0],
                          self.y.shape[1]))
        else:
            y_pred_means = \
                np.empty((self.n_posterior_samples, X.shape[0]))
        if eval_MSE:
            if self.y.shape[1] > 1:
                y_pred_variances = \
                    np.empty((self.n_posterior_samples, X.shape[0],
                              self.y.shape[1]))
            else:
                y_pred_variances = \
                    np.empty((self.n_posterior_samples, X.shape[0]))

        for i in range(self.n_posterior_samples):
            if eval_MSE:
                y_pred_means[i], y_pred_variances[i] = \
                    self.gps[i].predict(X, eval_MSE=True)
            else:
                y_pred_means[i] = self.gps[i].predict(X, eval_MSE=False)

        first_moments = y_pred_means.mean(0)
        if eval_MSE:
            second_moments = (first_moments**2 + y_pred_variances).mean(0)
            return first_moments, second_moments - first_moments**2
        else:
            return first_moments

    def lnpostfn(self, theta):
        """ Returns the log posterior probability of parameter theta. """
        theta = np.asarray(theta)
        if np.any((self.base_gp.thetaL * (1 - 1e-5) > theta)
                  + (theta > self.base_gp.thetaU * (1 + 1e-5))):
            return -np.inf
        log_prior = self.base_gp.corr.log_prior(theta)
        log_likelihood = self.base_gp.reduced_likelihood_function(theta)[0]
        return log_prior + log_likelihood

    def __getstate__(self):
        """ Return a pickable state for this object """
        odict = self.__dict__.copy()  # copy the dict since we change it
        odict.pop("sampler", None)
        return odict
