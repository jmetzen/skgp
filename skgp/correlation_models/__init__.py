
import stationary
import non_stationary

CORRELATION_TYPES = {
    'absolute_exponential': stationary.AbsoluteExponential,
    'squared_exponential': stationary.SquaredExponential,
    'matern_1.5': stationary.Matern_1_5,
    'matern_2.5': stationary.Matern_2_5,
    'generalized_exponential': stationary.GeneralizedExponential,
    'cubic': stationary.Cubic,
    'linear': stationary.Linear,
    'local_length_scales': non_stationary.LocalLengthScalesCorrelation,
    'manifold': non_stationary.ManifoldCorrelation}
