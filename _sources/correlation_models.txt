.. _correlation_models:

Correlation models
==================
One of the main components of skgp is an extended set of correlation models (kernels). These extended correlation models are dicussed in the following.

Distances
---------
skgp ships different models of distance that can be learned from data and that can be used within most of the kernels. Three different distance models, which differ in the number of free parameters and thus their capacity to adapt to the data, are provided:
 * *Isotropic*: a global length scale is learned from data. This means only one parameter is learned but also that different length scales in the different dimensions of the data cannot be identified and exploited.
 * *Automatic relevance determination (ARD)*: every dimension gets its own
   characteristic length scale, irrelevant dimensions can be effectively pruned away. This is also called "anisotropic" distance. The number of parameters is equal to the dimensionality of the data.
 * *Factor analysis distance (FAD)*: A low-rank approximation of a full
   covariance matrix for the distance is learned. Correlations between different dimensions can be identified and exploited to some extent. The number of free parameters corresponds to the number of dimensions times the desired rank of the matrix.

See Rasmussen and Williams 2006, p107 for details regarding the different
variants. In an experiment, we compare the three distances when combined with the squared exponential kernel. The target function maps a 4-dimensional vector onto a real value. One of
the dimensions is ignored (and should thus be pruned away by ARD). The other
dimensions are correlated, which can be exploited by FAD. Learning curves for different distance models are shown in the following figure:

.. figure:: _images/plot_gp_learning_curve_1.png
    :target: auto_examples/plot_gp_learning_curve.html
    :scale: 30%
    :align: center

As can be seen, in this example the more complex and flexible models perform considerably better. This need not always be the case as a more complex model might also overfit to the data at hand. However, if redundant or irrelevant dimensions are to be expected, a more adaptive distance measure typically pays off.


Matérn kernel
-------------

The class of `Matérn kernels
<http://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function>`_ is a
generalization of kernels like the squared or absolute exponential kernel. It
has an additional parameter :math:`\nu` which controls the smoothness of the
Gaussian process resulting from this kernel: a Gaussian process with Matérn
covariance has sample paths that are :math:`\lfloor \nu-1 \rfloor` times
differentiable. The general functional form of a Matérn for distance :math:`d`
is given by:

.. math::

    C(d) = \sigma^2\frac{1}{\Gamma(\nu)2^{\nu-1}}\Bigg(\sqrt{2\nu}\frac{d}{\rho}\Bigg)^\nu K_\nu\Bigg(\sqrt{2\nu}\frac{d}{\rho}\Bigg),

As :math:`\nu\rightarrow\infty`, the Matérn kernel converges to the squared
exponential kernel, i.e.,

.. math::
    C(d) = \sigma^2\exp(-d^2/ 2\rho^2).

When :math:`\nu = 1/2`, the Matérn kernel becomes identical to the absolute
exponential kernel, i.e.,

.. math::
    C(d) = \sigma^2 \exp \Bigg(-\frac{d}{\rho} \Bigg) \quad \quad \nu= \tfrac{1}{2}

Further important special cases are

.. math::
    C(d) = \sigma^2 \Bigg(1 + \frac{ \sqrt{3}d }{\rho} \Bigg) \exp \Bigg(-\frac{\sqrt{3}d}{\rho} \Bigg) \quad \quad \nu= \tfrac{3}{2}

and

.. math::
    C(d) = \sigma^2 \Bigg(1 + \frac{ \sqrt{5}d }{\rho} +\frac{ 5d^2}{3 \rho^2 }   \Bigg) \exp \Bigg(-\frac{\sqrt{5}d}{\rho} \Bigg) \quad \quad \nu= \tfrac{5}{2}.


See Rasmussen and Williams 2006, pp84 for further details regarding the
different variants of the Matérn kernel. In particular, :math:`\nu = 3/2` and
:math:`\nu = 5/2` are popular choices for learning functions that are not
infinitely differentiable (as assumed by the squared exponential) but at least
once (:math:`\nu = 3/2`) or twice differentiable (:math:`\nu = 5/2`). This is a
common case for real world data.

The following graphic illustrates learning curves for different variants of the
Matérn kernel. The target function is a twice differentiable function which is
created by fitting a spline to randomly sampled data over the interval
:math:`[0, 10]`:

.. figure:: _images/plot_matern_kernel_1.png
    :target: auto_examples/plot_matern_kernel.html
    :scale: 30%
    :align: center

As can be seen, neither the squared-exponential kernel (being too smooth) nor the absolute exponential kernel (being to rough) are good choices here. In contrast, the Matérn kernel with :math:`\nu = 5/2` (assuming correctly that the function is twice differentiable) performs considerably better. Also, the Matérn kernel with :math:`\nu = 3/2` is only slightly worse.

Non-stationary kernel
---------------------

.. figure:: _images/plot_gp_nonstationary_1.png
    :target: auto_examples/plot_gp_nonstationary.html
    :scale: 30%
    :align: center

.. figure:: _images/plot_gp_nonstationary_2.png
    :target: auto_examples/plot_gp_nonstationary.html
    :scale: 30%
    :align: center
