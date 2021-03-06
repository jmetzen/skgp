# -*- coding: utf-8 -*-

# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         (mostly translation, see implementation details)
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#         (converting to a object-oriented, more modular design)
# Licence: BSD 3 clause

"""
The built-in correlation models submodule for the gaussian_process module.
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.utils import check_array
from sklearn.externals.six import with_metaclass

MACHINE_EPSILON = np.finfo(np.double).eps


def l1_cross_differences(X):
    """
    Computes the nonzero componentwise differences between the vectors
    in X.

    Parameters
    ----------

    X: array_like
        An array with shape (n_samples, n_features)

    Returns
    -------

    D: array with shape (n_samples * (n_samples - 1) / 2, n_features)
        The array of componentwise differences.

    ij: arrays with shape (n_samples * (n_samples - 1) / 2, 2)
        The indices i and j of the vectors in X associated to the cross-
        distances in D: D[k] = np.abs(X[ij[k, 0]] - Y[ij[k, 1]]).
    """
    X = check_array(X)
    n_samples, n_features = X.shape
    n_nonzero_cross_diff = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_diff, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_diff, n_features))
    ll_1 = 0
    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = X[k] - X[(k + 1):n_samples]

    return D, ij.astype(np.int)


class StationaryCorrelation(with_metaclass(ABCMeta, object)):
    """ Base-class for stationary correlation models for Gaussian Processes.

    Stationary correlation models dependent only on the relative distance
    and not on the absolute positions of the respective datapoints. We can thus
    work internally solely on these distances.
    """
    def __init__(self):
        pass

    def fit(self, X, nugget=10. * MACHINE_EPSILON):
        """ Fits the correlation model for training data X

        Parameters
        ----------
        X : array_like, shape=(n_samples, n_features)
            An array of training datapoints at which observations were made,
            i.e., where the outputs y are known
        nugget : double or ndarray, optional
            The Gaussian Process nugget parameter
            The nugget is added to the diagonal of the assumed training
            covariance; in this way it acts as a Tikhonov regularization in
            the problem.  In the special case of the squared exponential
            correlation function, the nugget mathematically represents the
            variance of the input values. Default assumes a nugget close to
            machine precision for the sake of robustness
            (nugget = 10. * MACHINE_EPSILON).
        """
        self.X = X
        self.nugget = nugget
        self.n_samples = X.shape[0]

        # Calculate array with shape (n_eval, n_features) giving the
        # componentwise distances between locations x and x' at which the
        # correlation model should be evaluated.
        self.D, self.ij = l1_cross_differences(self.X)
        if (np.min(np.sum(self.D, axis=1)) == 0.
           and not isinstance(self, PureNugget)):
            raise Exception("Multiple input features cannot have the same"
                            " value.")

    def __call__(self, theta, X=None):
        """ Compute correlation for given correlation parameter(s) theta.

        Parameters
        ----------
        theta : array_like
            An array with giving the autocorrelation parameter(s).
            Dimensionality depends on the specific correlation model; often
            shape (1,) corresponds to an isotropic correlation model and shape
            (n_features,) to a anisotropic one.

        X : array_like, shape(n_eval, n_features)
            An array containing the n_eval query points whose correlation with
            the training datapoints shall be computed. If None, autocorrelation
            of the training datapoints is computed instead.

        Returns
        -------
        r : array_like, shape=(n_eval, n_samples) if X != None
                              (n_samples, n_samples) if X == None
            An array containing the values of the correlation model.
        """
        theta = np.asarray(theta, dtype=np.float)
        if X is not None:
            # Get pairwise componentwise L1-differences to the input training
            # set
            d = X[:, np.newaxis, :] - self.X[np.newaxis, :, :]
            d = d.reshape((-1, X.shape[1]))
        else:
            # No external datapoints given; auto-correlation of training set
            # is used instead
            d = self.D

        if d.ndim > 1:
            n_features = d.shape[1]
        else:
            n_features = 1

        # Compute the correlation for the respective correlation model (handled
        # by subclass)
        r = self._compute_corr(theta, d, n_features)

        if X is not None:
            # Convert to 2d matrix
            return r.reshape(-1, self.n_samples)
        else:
            # Auto-correlation computed only for upper triangular part of
            # matrix. Fill diagonal with 1+nugget and the lower triangular
            # by exploiting symmetry of matrix
            R = np.eye(self.n_samples) * (1. + self.nugget)
            R[self.ij[:, 0], self.ij[:, 1]] = r
            R[self.ij[:, 1], self.ij[:, 0]] = r
            return R

    def log_prior(self, theta):
        """ Returns the (log) prior probability of parameters theta.

        The prior is assumed to be uniform over the parameter space.
        NOTE: The returned quantity is an improper prior as its integral over
              the parameter space is not equal to 1.

        Parameters
        ----------
        theta : array_like, shape=(1,) or (n_features,)
            An array with shape 1 (isotropic) or n_features (anisotropic)
            giving the autocorrelation parameter(s).

        Returns
        -------
        log_p : float
            The (log) prior probability of parameters theta. An improper
            probability.
        """
        return 0

    @abstractmethod
    def _compute_corr(self, theta, d, n_features):
        """ Correlation for given pairwise, component-wise L1-differences.

        Parameters
        ----------
        theta : array_like, shape=(1,) or (n_features,)
            An array with shape 1 (isotropic) or n_features (anisotropic)
            giving the autocorrelation parameter(s).

        d : array_like, shape=(n_eval, n_features)
            An array with the pairwise, component-wise L1-differences of x
            and x' at which the correlation model should be evaluated.

        Returns
        -------
        r : array_like, shape=(n_eval, )
            An array containing the values of the autocorrelation model.
        """


class AbsoluteExponential(StationaryCorrelation):
    """ Absolute exponential autocorrelation model.

    Absolute exponential autocorrelation model (Ornstein-Uhlenbeck stochastic
    process)::

                                              n
            theta, d --> r(theta, d) = exp(  sum  - theta_i * d_i )
                                             i = 1
    """

    def _compute_corr(self, theta, d, n_features):
        """ Correlation for given pairwise, component-wise L1-differences.

        Parameters
        ----------
        theta : array_like, shape=(1,) or (n_features,)
            An array with shape 1 (isotropic) or n_features (anisotropic)
            giving the autocorrelation parameter(s).

        d : array_like, shape=(n_eval, n_features)
            An array with the pairwise, component-wise L1-differences of x
            and x' at which the correlation model should be evaluated.

        Returns
        -------
        r : array_like, shape=(n_eval, )
            An array containing the values of the autocorrelation model.
        """
        d = np.asarray(d, dtype=np.float)
        d = np.abs(d)
        if theta.size == 1:
            return np.exp(- theta[0] * np.sum(d, axis=1))
        elif theta.size != n_features:
            raise ValueError("Length of theta must be 1 or %s" % n_features)
        else:
            return np.exp(- np.sum(theta.reshape(1, n_features) * d, axis=1))


class SquaredExponential(StationaryCorrelation):
    """ Squared exponential correlation model.

    Squared exponential correlation model (Radial Basis Function).
    (Infinitely differentiable stochastic process, very smooth)::

                                              n
            theta, d --> r(theta, d) = exp(  sum  - theta_i * (d_i)^2 )
                                            i = 1
    """

    def _compute_corr(self, theta, d, n_features):
        """ Correlation for given pairwise, component-wise L1-differences.

        Parameters
        ----------
        theta : array_like, shape=(1,) [isotropic]
                                  (n_features,) [anisotropic] or
                                  (k*n_features,) [factor analysis distance]
            An array encoding the autocorrelation parameter(s).

        d : array_like, shape=(n_eval, n_features)
            An array with the pairwise, component-wise L1-differences of x
            and x' at which the correlation model should be evaluated.

        Returns
        -------
        r : array_like, shape=(n_eval, )
            An array containing the values of the autocorrelation model.
        """
        d = np.asarray(d, dtype=np.float)
        return np.exp(-self._quadratic_activation(theta, d, n_features))

    def _quadratic_activation(self, theta, d, n_features):
        """ Utility function for computing quadratic activation.

        Computes the activation activ=d.T * M * d where M is a covariance
        matrix of size n*n. The hyperparameters theta specify
         * an isotropic covariance matrix, i.e., M = theta * I with I being the
           identity, if theta has shape 1
         * an automatic relevance determination model if theta has shape n,
           in which the characteristic length scales of each dimension are
           learned separately:  M = diag(theta)
         * a factor analysis distance model if theta has shape k*n for k> 1,
           in which a low-rank approximation of the full matrix M is learned.
           This low-rank approximation approximates the covariance matrix as
           low-rank matrix plus a diagonal matrix:
           M = Lambda * Lambda.T + diag(l),
           where Lambda is a n*(k-1) matrix and l specifies the diagonal
           matrix.

        Parameters
        ----------
        theta : array_like, shape=(1,) [isotropic]
                                  (n_features,) [anisotropic] or
                                  (k*n_features,) [factor analysis distance]
            An array encoding the autocorrelation parameter(s). In the
            case of the factor analysis distance, M is approximated by
            M = Lambda * Lambda.T + diag(l), where l is encoded in the last n
            entries of theta and Lambda is encoded row-wise in the first
            entries of theta. Note that Lambda may contain negative entries
            while theta is strictly positive; because of this, the entries of
            Lambda are set to the logarithm with basis 10 of the corresponding
            entries in theta.

        array_like, shape=(n_eval, n_features)
            An array giving the componentwise differences of x and x' at
            which the quadratic activation should be evaluated.

        Returns
        -------
        a : array_like, shape=(n_eval, )
            An array with the activation values for the respective
            componentwise differences d.
        """
        if theta.size == 1:  # case where M is isotropic: M = diag(theta[0])
            return theta[0] * np.sum(d ** 2, axis=1)
        elif theta.size == n_features:  # anisotropic but diagonal case (ARD)
            return np.sum(theta.reshape(1, n_features) * d ** 2, axis=1)
        elif theta.size % n_features == 0:
            # Factor analysis case: M = lambda*lambda.T + diag(l)
            theta = theta.reshape((1, theta.size))
            M = np.diag(theta[0, :n_features])  # the diagonal matrix part l
            # The low-rank matrix contribution which allows accounting for
            # correlations in the feature dimensions
            # NOTE: these components of theta are passed through a log-function
            #       to allow negative values in Lambda
            Lambda = np.log10(theta[0, n_features:].reshape((n_features, -1)))
            M += Lambda.dot(Lambda.T)
            return np.sum(d.dot(M) * d, -1)
        else:
            raise ValueError("Length of theta must be 1 or a multiple of %s."
                             % n_features)


class Matern_1_5(SquaredExponential):
    """ Matern correlation model for nu=1.5.

    Sample paths are once differentiable. Given by::

        r(theta, dx) = (1 + np.sqrt(3*activ))*exp(-np.sqrt(3*activ))
    where activ=dx.T * M * dx and M is a covariance matrix of size n*n.

    See Rasmussen and Williams 2006, pp84 for details regarding the different
    variants of the Matern kernel.
    """

    def _compute_corr(self, theta, d, n_features):
        """ Correlation for given pairwise, component-wise L1-differences.

        Parameters
        ----------
        theta : array_like, shape=(1,) [isotropic]
                                  (n_features,) [anisotropic] or
                                  (k*n_features,) [factor analysis distance]
            An array encoding the autocorrelation parameter(s).

        d : array_like, shape=(n_eval, n_features)
            An array with the pairwise, component-wise L1-differences of x
            and x' at which the correlation model should be evaluated.

        Returns
        -------
        r : array_like, shape=(n_eval, )
            An array containing the values of the autocorrelation model.
        """
        d = np.asarray(d, dtype=np.float)
        activ = self._quadratic_activation(theta, d, n_features)
        tmp = np.sqrt(3 * activ)  # temporary variable for preventing
                                  # recomputation
        return (1 + tmp) * np.exp(-tmp)


class Matern_2_5(SquaredExponential):
    """ Matern correlation model for nu=2.5.

    Sample paths are twice differentiable. Given by::

       r(theta, dx) = (1 + np.sqrt(5*activ) + 5/3*activ)*exp(-np.sqrt(5*activ))
    where activ=dx.T * M * dx and M is a covariance matrix of size n*n.

    See Rasmussen and Williams 2006, pp84 for details regarding the different
    variants of the Matern kernel.
    """

    def _compute_corr(self, theta, d, n_features):
        """ Correlation for given pairwise, component-wise L1-differences.

        Parameters
        ----------
        theta : array_like, shape=(1,) [isotropic]
                                  (n_features,) [anisotropic] or
                                  (k*n_features,) [factor analysis distance]
            An array encoding the autocorrelation parameter(s).

        d : array_like, shape=(n_eval, n_features)
            An array with the pairwise, component-wise L1-differences of x
            and x' at which the correlation model should be evaluated.

        Returns
        -------
        r : array_like, shape=(n_eval, )
            An array containing the values of the autocorrelation model.
        """
        d = np.asarray(d, dtype=np.float)
        activ = self._quadratic_activation(theta, d, n_features)
        tmp = np.sqrt(5 * activ)  # temporary variable for preventing
                                  # recomputation
        return (1 + tmp + 5.0 / 3.0 * activ) * np.exp(-tmp)


class GeneralizedExponential(StationaryCorrelation):
    """ Generalized exponential correlation model.

    Generalized exponential correlation model.
    (Useful when one does not know the smoothness of the function to be
    predicted.)::

                                          n
        theta, d --> r(theta, d) = exp(  sum  - theta_i * |d_i|^p )
                                        i = 1
    """

    def _compute_corr(self, theta, d, n_features):
        """ Correlation for given pairwise, component-wise L1-differences.

        Parameters
        ----------
        theta : array_like, shape=(1+1,) or (n_features+1,)
            An array with shape 1+1 (isotropic) or n_features+1 (anisotropic)
            giving the autocorrelation parameter(s) (theta, p).

        d : array_like, shape=(n_eval, n_features)
            An array with the pairwise, component-wise L1-differences of x
            and x' at which the correlation model should be evaluated.

        Returns
        -------
        r : array_like, shape=(n_eval, )
            An array containing the values of the autocorrelation model.
        """
        d = np.asarray(d, dtype=np.float)
        lth = theta.size
        if n_features > 1 and lth == 2:
            theta = np.hstack([np.repeat(theta[0], n_features), theta[1]])
        elif lth != n_features + 1:
            raise Exception("Length of theta must be 2 or %s"
                            % (n_features + 1))
        else:
            theta = theta.reshape(1, lth)

        td = theta[:, 0:-1].reshape(1, n_features) \
            * np.abs(d) ** theta[:, -1]
        return np.exp(- np.sum(td, 1))


class PureNugget(StationaryCorrelation):
    """ Spatial independence correlation model (pure nugget).

    Useful when one wants to solve an ordinary least squares problem!::

                                                n
            theta, d --> r(theta, dx) = 1 if   sum |d_i| == 0
                                              i = 1
                                        0 otherwise
    """

    def _compute_corr(self, theta, d, n_features):
        """ Correlation for given pairwise, component-wise L1-differences.

        Parameters
        ----------
        theta : array_like
            None.

        d : array_like, shape=(n_eval, n_features)
            An array with the pairwise, component-wise L1-differences of x
            and x' at which the correlation model should be evaluated.


        Returns
        -------
        r : array_like
            An array with shape (n_eval, ) with the values of the
            autocorrelation model.
        """
        d = np.asarray(d, dtype=np.float)
        n_eval = d.shape[0]
        r = np.zeros(n_eval)
        r[np.all(d == 0., axis=1)] = 1.

        return r


class Cubic(StationaryCorrelation):
    """ Cubic correlation model.

    Cubic correlation model::

        theta, d --> r(theta, d) =
          n
        prod max(0, 1 - 3(theta_j*d_ij)^2 + 2(theta_j*d_ij)^3) ,  i = 1,...,m
        j = 1
    """

    def _compute_corr(self, theta, d, n_features):
        """ Correlation for given pairwise, component-wise L1-differences.

        Parameters
        ----------
        theta : array_like, shape=(1,) or (n_features,)
            An array with shape 1 (isotropic) or n_features (anisotropic)
            giving the autocorrelation parameter(s).

        d : array_like, shape=(n_eval, n_features)
            An array with the pairwise, component-wise L1-differences of x
            and x' at which the correlation model should be evaluated.

        Returns
        -------
        r : array_like, shape=(n_eval, )
            An array containing the values of the autocorrelation model.
        """
        d = np.asarray(d, dtype=np.float)
        lth = theta.size
        if lth == 1:
            td = np.abs(d) * theta
        elif lth != n_features:
            raise Exception("Length of theta must be 1 or " + str(n_features))
        else:
            td = np.abs(d) * theta.reshape(1, n_features)

        td[td > 1.] = 1.
        ss = 1. - td ** 2. * (3. - 2. * td)
        return np.prod(ss, 1)


class Linear(StationaryCorrelation):
    """ Linear correlation model.

    Linear correlation model::

        theta, d --> r(theta, d) =
              n
            prod max(0, 1 - theta_j*d_ij) ,  i = 1,...,m
            j = 1
    """

    def _compute_corr(self, theta, d, n_features):
        """ Correlation for given pairwise, component-wise L1-differences.

        Parameters
        ----------
        theta : array_like, shape=(1,) or (n_features,)
            An array with shape 1 (isotropic) or n_features (anisotropic)
            giving the autocorrelation parameter(s).

        d : array_like, shape=(n_eval, n_features)
            An array with the pairwise, component-wise L1-differences of x
            and x' at which the correlation model should be evaluated.

        Returns
        -------
        r : array_like, shape=(n_eval, )
            An array containing the values of the autocorrelation model.
        """
        d = np.asarray(d, dtype=np.float)
        lth = theta.size
        if lth == 1:
            td = np.abs(d) * theta
        elif lth != n_features:
            raise Exception("Length of theta must be 1 or %s" % n_features)
        else:
            td = np.abs(d) * theta.reshape(1, n_features)

        td[td > 1.] = 1.
        ss = 1. - td
        return np.prod(ss, 1)
