""" Non-stationary correlation models for Gaussian processes."""
# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import numpy as np
from scipy.special import gamma, kv
from scipy.stats import expon, norm

from sklearn.cluster import KMeans

from skgp.correlation_models.stationary import l1_cross_differences

MACHINE_EPSILON = np.finfo(np.double).eps


class LocalLengthScalesCorrelation(object):
    """ Non-stationary correlation model based on local smoothness estimates.

    This non-stationary correlation model learns internally point estimates of
    local smoothness using a second-level Gaussian Process. For this, it
    selects a subset of the training data and learns length-scales at this
    specific points. These length scales are generalized using the second-level
    Gaussian Process. Furthermore, global (isotropic or anisotropic) length
    scales are learned for both the top-level GP and the length-scale GP.

    The correlation model is based on the family of (stationary) Matern
    kernels. The parameter nu of the Matern kernels (governing the smoothness
    of the GP prior) can either be set or learned jointly with the remaining
    parameters.

    Parameters
    ----------
    isotropic : bool, default=True
        Whether the global length-scales of the top-level GP are isotropic or
        anisotropic

    nu: float, default=1.5
        The parameter nu of the Matern kernels (governing the smoothness
        of the GP prior). If None, nu is learned along with the other
        hyperparameters.

    l_isotropic : bool, default=True
        Whether the global length-scales of the length-scale GP are isotropic
        or anisotropic

    l_samples: int, default=10
        How many datapoints from the training data are selected as support
        points for learning the length-scale GP

    prior_b: float, default=inf
        The variance of the log-normal prior distribution on the length scales.
        If set to infinity, the distribution is assumed to be uniform.

    .. seealso::

    "Nonstationary Gaussian Process Regression using Point Estimates of Local
    Smoothness", Christian Plagemann, Kristian Kersting, and Wolfram Burgard,
    ECML 2008
    """

    def __init__(self, isotropic=True, nu=1.5, l_isotropic=True, l_samples=10,
                 prior_b=np.inf, X_=None):
        self.isotropic = isotropic
        self.nu = nu
        self.l_isotropic = l_isotropic
        self.l_samples = l_samples
        self.prior_b = prior_b
        self.X_ = X_
        if self.X_ is not None:
            assert self.X_.shape[0] == self.l_samples

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
        self.n_dims = X.shape[1]

        # Determine how many entries in theta belong to the different
        # categories (used later for parsing theta)
        self.theta_gp_size = 1 if self.isotropic else self.n_dims
        self.theta_l_size = 1 if self.l_isotropic else self.n_dims
        self.nu_size = 1 if not self.nu else 0
        self.theta_size = self.theta_gp_size + self.theta_l_size \
            + self.l_samples + self.nu_size

        # Calculate array with shape (n_eval, n_features) giving the
        # componentwise distances between locations x and x' at which the
        # correlation model should be evaluated.
        self.D, self.ij = l1_cross_differences(self.X)

        if self.X_ is None:
            # Select subset of X for which length scales are optimized.
            # Generalization of length scales to other datapoints is acheived
            # by means of a separate Gaussian Process (gp_l)
            if self.X.shape[0] >= self.l_samples:
                kmeans = KMeans(n_clusters=self.l_samples)
                self.X_ = kmeans.fit(self.X).cluster_centers_
            else:  # Fallback to select centers using sampling with replacement
                self.X_ = self.X[np.random.choice(np.arange(self.X.shape[0]),
                                                  self.l_samples)]

        return self

    def __call__(self, theta, X=None):
        """ Compute correlation for given correlation parameter(s) theta.

        Parameters
        ----------
        theta : array_like
            An array giving the autocorrelation parameter(s).

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
        # Parse theta into its components
        theta_gp, theta_l, length_scales, nu = self._parse_theta(theta)

        # Train length-scale Gaussian Process
        from skgp.estimators import GaussianProcess
        self.gp_l = \
            GaussianProcess(corr="matern_1.5",
                            theta0=theta_l).fit(self.X_,
                                                np.log10(length_scales))
        l_train = 10**self.gp_l.predict(self.X)

        # Prepare distances and length scale information for any pair of
        # datapoints, whose correlation shall be computed
        if X is not None:
            # Get pairwise componentwise L1-differences to the input training
            # set
            d = X[:, np.newaxis, :] - self.X[np.newaxis, :, :]
            d = d.reshape((-1, X.shape[1]))
            # Predict length scales for query datapoints
            l_query = 10**self.gp_l.predict(X)
            l = np.transpose([np.tile(l_train, len(l_query)),
                              np.repeat(l_query, len(l_train))])
        else:
            # No external datapoints given; auto-correlation of training set
            # is used instead
            d = self.D
            l = l_train[self.ij]

        # Compute general Matern kernel
        if d.ndim > 1 and theta_gp.size == d.ndim:
            activation = np.sum(theta_gp.reshape(1, d.ndim) * d ** 2, axis=1)
        else:
            activation = theta_gp[0] * np.sum(d ** 2, axis=1)
        tmp = 0.5*(l**2).sum(1)
        tmp2 = np.maximum(2*np.sqrt(nu * activation / tmp), 1e-5)
        r = np.sqrt(l[:, 0]) * np.sqrt(l[:, 1]) / (gamma(nu) * 2**(nu - 1))
        r /= np.sqrt(tmp)
        r *= tmp2**nu * kv(nu, tmp2)

        # Convert correlations to 2d matrix
        if X is not None:
            return r.reshape(-1, self.n_samples)
        else:  # exploit symmetry of auto-correlation
            R = np.eye(self.n_samples) * (1. + self.nugget)
            R[self.ij[:, 0], self.ij[:, 1]] = r
            R[self.ij[:, 1], self.ij[:, 0]] = r
            return R

    def log_prior(self, theta):
        """ Returns the (log) prior probability of parameters theta.

        The prior is assumed to be uniform over the parameter space except for
        the length-scales dimensions. These are assumed to be log-normal
        distributed with mean 0 and variance self.prior_b. If
        self.prior_b is np.inf, the log length-scales are assumed to be
        uniformly distributed as well.

        NOTE: The returned quantity is an improper prior as its integral over
              the parameter space is not equal to 1.

        Parameters
        ----------
        theta : array_like
            An array giving the autocorrelation parameter(s).

        Returns
        -------
        log_p : float
            The (log) prior probability of parameters theta. An improper
            probability.
        """
        if self.prior_b == np.inf:
            return 0.0
        _, _, length_scales, _ = self._parse_theta(theta)

        squared_dist = (np.log10(length_scales)**2).sum()
        return -squared_dist / self.prior_b

    def _parse_theta(self, theta):
        """ Parse parameter vector theta into its components.

        Parameters
        ----------
        theta : array_like
            An array containing all hyperparameters.

        Returns
        -------
        theta_gp : array_like
            An array containing the hyperparameters of the main GP.
        theta_l : array_like
            An array containing the hyperparameters of the length-scale GP.
        length_scales : array_like
            An array containing the length-scales for the length-scale GP.
        nu : float
            The parameter nu controlling the smoothness of the Matern kernel.
        """
        theta = np.asarray(theta, dtype=np.float)

        assert (theta.size == self.theta_size), \
            "theta does not have the expected size (expected: %d, " \
            "actual size %d). Expected: %d entries for main GP, " \
            "%d entries for length-scale GP, %d entries containing the "\
            "length scales, and %d entries for nu." \
            % (self.theta_size, theta.size, self.theta_gp_size,
               self.theta_l_size, self.l_samples, self.nu_size)

        # Split theta in its components
        theta_gp = theta[:self.theta_gp_size]
        theta_l = \
            theta[self.theta_gp_size:][:self.theta_l_size]
        length_scales = \
            theta[self.theta_gp_size+self.theta_l_size:][:self.l_samples]
        nu = self.nu if self.nu else theta[-1]

        return theta_gp, theta_l, length_scales, nu

    @classmethod
    def create(cls, dims, isotropic=True, theta0=1e-1,
               thetaL=None, thetaU=None,
               l_isotropic=True, theta_l_0=1e-1,
               theta_l_L=None, theta_l_U=None,
               l_samples=20, l_0=1.0, l_L=None, l_U=None,
               nu_0=1.5, nu_L=None, nu_U=None, prior_b=np.inf,
               *args, **kwargs):
        """ Factory method for creating non-stationary correlation models.

        ..note:: In addtion to returning an instance of
                 NonStationaryCorrelation, the specification of the search
                 space for the hyperparameters theta of the Gaussian process
                 is returned. This includes the start point of the search
                 (theta0) as well as the lower and upper boundaries thetaL and
                 thetaU for the values of theta.
        """
        theta0 = [theta0] * (1 if isotropic else dims)
        thetaL = [thetaL] * (1 if isotropic else dims)
        thetaU = [thetaU] * (1 if isotropic else dims)

        theta0 += [theta_l_0] * (1 if l_isotropic else dims)
        thetaL += [theta_l_L] * (1 if l_isotropic else dims)
        thetaU += [theta_l_U] * (1 if l_isotropic else dims)

        theta0 += [l_0] * l_samples
        thetaL += [l_L] * l_samples
        thetaU += [l_U] * l_samples

        if nu_L is not None:
            theta0 += [nu_0]
            thetaL += [nu_L]
            thetaU += [nu_U]

        corr = cls(isotropic=isotropic, nu=None if nu_L else nu_0,
                   l_isotropic=l_isotropic, l_samples=l_samples,
                   prior_b=prior_b)

        return corr, theta0, thetaL, thetaU


class ManifoldCorrelation(object):
    """ Non-stationary correlation model based on manifold learning.

    This non-stationary correlation model consists internally of two parts:
    a mapping from the actual data space onto a manifold and a stationary
    correlation model on this manifold. The mapping is realized by a neural
    network whose architecture can be specified externally. The parameters of
    this network are learned along with the length scales of the Gaussian
    process, typically such that the marginal likelihood or the posterior
    probability of the GP are maximized. Any common stationary correlation
    model can then be used on top of this manifold.

    Parameters
    ----------
    base_corr: string or instance of StationaryCorrelation, optional
        The top-level, stationary autocorrelation function returning
        the autocorrelation between two points M(x) and M(x') on the manifold.
        Built-in correlation models are::

            'absolute_exponential', 'squared_exponential',
            'generalized_exponential', 'cubic', 'linear'

    architecture: sequence of tuples
        Defines the structure of the internal neural network architecture
        mapping the data from the original data space onto a manifold. Note
        that different data dimensions can be processed by different networks
        and that the networks can have different number of layers. For
        instance, the architecture ((1, 2),(2, 4, 5)) would map a 3-dimensional
        input space onto a 7-dimensional manifold. For this, the first input
        dimension would be processed by the network (1, 2) with 1 inputs,
        2 outputs, and no hidden layer yielding the first two manifold
        dimensions. The other two input dimensions would be processed by a
        network (2, 4, 5) with 2 inputs, 4 hidden units, and 5 outputs
        yielding the remaining five manifold dimensions.

    isotropic : bool, default=True
        Whether the global length-scales of the GP are isotropic or anisotropic

    prior_nn_scale: float, default=inf
        The standard deviation of the Gaussian prior distribution on the
        network parameters. If set to infinity, the distribution is assumed to
        be uniform.

    prior_gp_scale: float, default=inf
        The scale parameter of the exponential prior distribution on the
        length-scales. If set to infinity, the distribution is assumed to be
        uniform.

    transfer_fct: str, default="tanh"
        The transfer function used in the hidden and output units. Supported
        are "tanh" and the rectified linear unit ("relu"). Defaults is "tanh"

    .. seealso::

    "Manifold Gaussian Process for Regression",
    Roberto Calandra, Jan Peters, Carl Edward Rasmussen, Marc Peter Deisenroth,
    http://arxiv.org/abs/1402.5876
    """

    def __init__(self, base_corr, architecture, theta_nn_size,
                 isotropic=True, prior_nn_scale=np.inf, prior_gp_scale=np.inf,
                 transfer_fct="tanh"):
        self.architecture = architecture
        self.n_inputs = sum([subnet[0] for subnet in architecture])
        self.n_outputs = sum([subnet[-1] for subnet in architecture])
        self.theta_nn_size = theta_nn_size
        self.isotropic = isotropic
        self.prior_nn_scale = prior_nn_scale
        self.prior_gp_scale = prior_gp_scale
        self.transfer_fct = transfer_fct

        self.theta_gp_size = 1 if self.isotropic else self.n_outputs
        self.theta_size = self.theta_gp_size + self.theta_nn_size

        self.base_corr = base_corr
        if not callable(self.base_corr):
            from skgp.correlation_models import CORRELATION_TYPES
            if self.base_corr in CORRELATION_TYPES:
                self.base_corr = CORRELATION_TYPES[self.base_corr]()
            else:
                raise ValueError("base_corr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._correlation_types.keys(),
                                    self.base_corr))

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
        assert X.shape[1] == self.n_inputs

        self.X = X
        self.nugget = nugget
        self.n_samples = X.shape[0]

        return self

    def __call__(self, theta, X=None):
        """ Compute correlation for given correlation parameter(s) theta.

        Parameters
        ----------
        theta : array_like
            An array giving the autocorrelation parameter(s).

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
        # Parse theta into its components
        theta_gp, theta_nn = self._parse_theta(theta)

        # Map training data onto manifold
        if np.any(theta_nn == 0):
            theta_nn[np.where(theta_nn == 0)] \
                += np.random.random((theta_nn == 0).sum()) * 2e-5 - 1e-5
        X_train_nn = self._project_manifold(self.X, theta_nn)

        self.base_corr.fit(X_train_nn, nugget=self.nugget)

        if X is not None:
            X_test_nn = self._project_manifold(X, theta_nn)
            return self.base_corr(theta_gp, X_test_nn)
        else:
            return self.base_corr(theta_gp)

    def _project_manifold(self, X, theta_nn):
        # Lazila fetch transfer function (to keep object pickable)
        if self.transfer_fct == "tanh":
            transfer_fct = np.tanh
        elif self.transfer_fct == "sin":
            transfer_fct = np.sin
        elif self.transfer_fct == "relu":
            transfer_fct = lambda x: np.maximum(0, x)
        elif hasattr(self.transfer_fct, "__call__"):
            transfer_fct = self.transfer_fct

        y = []
        for subnet in self.architecture:
            y.append(X[:, :subnet[0]])
            for layer in range(len(subnet) - 1):
                W = theta_nn[:subnet[layer]*subnet[layer+1]]
                W = W.reshape((subnet[layer], subnet[layer+1]))
                b = theta_nn[subnet[layer]*subnet[layer+1]:
                             (subnet[layer]+1)*subnet[layer+1]]
                a = y[-1].dot(W) + b
                y[-1] = transfer_fct(a)

                # chop off weights of this layer
                theta_nn = theta_nn[(subnet[layer]+1)*subnet[layer+1]:]

            X = X[:, subnet[0]:]  # chop off used input dimensions

        return np.hstack(y)

    def log_prior(self, theta):
        """ Returns the (log) prior probability of parameters theta.

        TODO

        NOTE: The returned quantity is an improper prior as its integral over
              the parameter space is not equal to 1.

        Parameters
        ----------
        theta : array_like
            An array giving the autocorrelation parameter(s).

        Returns
        -------
        log_p : float
            The (log) prior probability of parameters theta. An improper
            probability.
        """
        theta_gp, theta_nn = self._parse_theta(theta)

        if self.prior_nn_scale == np.inf:
            prior_nn = 0.0
        else:
            prior_nn = norm.logpdf(theta_nn, scale=self.prior_nn_scale).sum()

        if self.prior_gp_scale == np.inf:
            prior_gp = 0.0
        else:
            prior_gp = expon.logpdf(theta_gp, scale=self.prior_gp_scale).sum()

        return prior_nn + prior_gp

    def _parse_theta(self, theta):
        """ Parse parameter vector theta into its components.

        Parameters
        ----------
        theta : array_like
            An array containing all hyperparameters.

        Returns
        -------
        theta_gp : array_like
            An array containing the hyperparameters of the main GP.
        theta_nn : array_like
            An array containing the hyperparameters of the manifold model.
        """
        theta = np.asarray(theta, dtype=np.float)

        assert (theta.size == self.theta_size), \
            "theta does not have the expected size (expected: %d, " \
            "actual size %d). Expected: %d entries for main GP and " \
            "%d entries for length-scale GP." \
            % (self.theta_size, theta.size, self.theta_gp_size,
               self.theta_nn_size)

        # Split theta in its components
        theta_gp = theta[:self.theta_gp_size]
        theta_nn = theta[self.theta_gp_size:]
        return theta_gp, np.log10(theta_nn)

    @classmethod
    def create(cls, base_corr, architecture, isotropic=True,
               theta0=1e-1, thetaL=None, thetaU=None,
               max_nn_weight=5, prior_nn_scale=np.inf, prior_gp_scale=np.inf,
               transfer_fct="tanh", *args, **kwargs):
        """ Factory method for creating manifold correlation models.

        ..note:: In addition to returning an instance of
                 ManifoldCorrelation, the specification of the search
                 space for the hyperparameters theta of the Gaussian process
                 is returned. This includes the start point of the search
                 (theta0) as well as the lower and upper boundaries thetaL and
                 thetaU for the values of theta.
        """
        assert "prior_b" not in kwargs
        n_outputs, theta_nn_size = cls.determine_network_layout(architecture)

        theta0 = [theta0] * (1 if isotropic else n_outputs)
        thetaL = [thetaL] * (1 if isotropic else n_outputs)
        thetaU = [thetaU] * (1 if isotropic else n_outputs)

        theta0 += \
            list(10**np.random.uniform(-max_nn_weight, max_nn_weight,
                                       theta_nn_size))
        thetaL += [10**-max_nn_weight] * theta_nn_size
        thetaU += [10**max_nn_weight] * theta_nn_size

        corr = cls(base_corr, architecture, theta_nn_size=theta_nn_size,
                   isotropic=isotropic, prior_nn_scale=prior_nn_scale,
                   prior_gp_scale=prior_gp_scale, transfer_fct=transfer_fct)

        return corr, theta0, thetaL, thetaU

    @staticmethod
    def determine_network_layout(architecture):
        """ Determine number of outputs and params of given architecture."""
        n_outputs = 0
        n_params = 0
        for subnet in architecture:
            for layer in range(len(subnet) - 1):
                n_params += (subnet[layer] + 1) * subnet[layer+1]

            n_outputs += subnet[-1]

        return n_outputs, n_params
