"""Basic probability distributions."""

import math
import numpy as np
from scipy.stats import lognorm, loguniform, uniform


class Uniform(object):
    """A uniform distribution.

    Attributes:
        lower_bound: [float] Lower bound for support.
        upper_bound: [float] Upper bound for support.
    """

    def __init__(self, lower_bound, upper_bound):
        """Initialises Uniform with bounds and sampler."""
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dist = uniform(lower_bound, upper_bound)

    def in_bound(self, value):
        """Check if value is in support."""
        if self.lower_bound <= value <= self.upper_bound: return 1
        else: return 0

    def log_pdf(self, value):
        """Calculate log probability density."""
        return self.dist.logpdf(value)


class Log_Uniform(object):
    """A log uniform distribution.

    The log of the data is uniformly distributed.

    Attributes:
        lower_bound: [float] Lower bound for support.
        upper_bound: [float] Upper bound for support.
    """

    def __init__(self, lower_bound, upper_bound):
        """Initialises Log uniform with bounds and sampler."""
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dist = loguniform(lower_bound, upper_bound)

    def in_bound(self, value):
        """Check if value is in support."""
        if self.lower_bound <= value <= self.upper_bound: return 1
        else: return 0

    def log_pdf(self, value):
        """Calculate log probability density."""
        return self.dist.logpdf(value)



class Truncated_Log_Normal(object):
    """A truncated log normal distribution.

    The log of the data is normally distributed, and the data is constrained.

    Attributes:
        lower_bound: [float] Lower bound for support.
        upper_bound: [float] Upper bound for support.
    """

    def __init__(self, lower_bound, upper_bound, mu, sd):
        """Initialises Truncated log normal with bounds and sampler.

        Args:
            mu: [float] The scalar mean of the underlying normal distrubtion in 
                true space.
            sd: [float] The scalar standard deviation of the underlying normal 
                distribution in true space.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dist = lognorm(scale = np.exp(np.log(mu)), s = (np.log(sd))) # Scipy shape parameters.

        # Probability that is otherwise truncated to zero, distributed uniformly (aprroximation).
        self.truncation = (self.dist.cdf(lower_bound) + 1 - self.dist.cdf(upper_bound)) / (upper_bound - lower_bound)

    def in_bound(self, value):
        """Check if value is in support."""
        if self.lower_bound <= value <= self.upper_bound: return 1
        else: return 0

    def log_pdf(self, value):
        """Calculate log probability density."""
        if self.lower_bound <= value <= self.upper_bound: return np.log(self.dist.pdf(value) + self.truncation)
        else: return -math.inf # If value is out of support.