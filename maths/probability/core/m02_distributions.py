from __future__ import annotations
import itertools
from typing import Any, Callable, Dict, Iterable, Optional
from abc import ABC, abstractmethod
import math
from maths.probability.core.m00_probability_foundations import ProbabilityMeasure, SampleSpace, Event, RandomVariable

# This class contains methods for working with probability distributions.
# It includes methods for calculating the mean, variance, and standard deviation
# of a distribution, as well as methods for generating random samples from
# various distributions.

# Abstract Base Class for Distributions
# Abstract base class for discrete probability distributions.
class DiscreteDistribution(ABC):
    @abstractmethod
    def pmf(self, k: int) -> float:
        """Probability Mass Function: P(X=k)"""
        pass

    @abstractmethod
    def cdf(self, k: int) -> float:
        """Cumulative Distribution Function: P(X<=k)"""
        pass

    @abstractmethod
    def expected_value(self) -> float:
        """Expected Value (Mean) of the distribution"""
        pass

    @abstractmethod
    def variance(self) -> float:
        """Variance of the distribution"""
        pass

    @abstractmethod
    def std_dev(self) -> float:
        """Standard Deviation of the distribution"""
        return math.sqrt(self.variance())

# Abstract class for continuous probability distributions
class ContinuousDistribution(ABC):
    @abstractmethod
    def pdf(self, x: float) -> float:
        """Probability Density Function: f(x)"""
        pass

    @abstractmethod
    def cdf(self, x: float) -> float:
        """Cumulative Distribution Function: F(x)"""
        pass

    @abstractmethod
    def expected_value(self) -> float:
        """Expected Value (Mean) of the distribution"""
        pass

    @abstractmethod
    def variance(self) -> float:
        """Variance of the distribution"""
        pass

    @abstractmethod
    def std_dev(self) -> float:
        """Standard Deviation of the distribution"""
        return math.sqrt(self.variance())

    @staticmethod
    def _integrate_simpson(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
        """Numerical integration using Simpson's rule."""
        if n % 2 == 1:
            n += 1  # n must be even
        h = (b - a) / n
        integral = f(a) + f(b)
        for i in range(1, n, 2):
            integral += 4 * f(a + i * h)
        for i in range(2, n-1, 2):
            integral += 2 * f(a + i * h)
        return integral * h / 3 

# Discrete Distributions
# Bernoulli Distribution: Single trial with success probability p.
class BernoulliDistribution(DiscreteDistribution):
    def __init__(self, p: float):
        if not(0.0 <= p <= 1.0):
            raise ValueError("Probability p must be in between 0 and 1.")
        self.p = p

    def pmf(self, k: int) -> float:
        if k not in (0, 1):
            raise ValueError("k must be 0 or 1 for Bernoulli distribution.")
        return self.p if k == 1 else (1 - self.p)

    def cdf(self, k: int) -> float:
        if k < 0:
            return 0.0
        elif k < 1:
            return 1 - self.p
        else:
            return 1.0

    def expected_value(self) -> float:
        return self.p

    def variance(self) -> float:
        return self.p * (1 - self.p) 

    def std_dev(self) -> float:
        return math.sqrt(self.variance())       

# Binomial Distribution: Number of successes in n independent Bernoulli trials.
class BinomialDistribution(DiscreteDistribution):
    def __init__(self, n: int, p: float):
        if n < 1:
            raise ValueError("Number of trials n must be at least 1.")
        if not(0.0 <= p <= 1.0):
            raise ValueError("Probability p must be in between 0 and 1.")
        self.n = n
        self.p = p

    def pmf(self, k: int) -> float:
        if k < 0 or k > self.n or not isinstance(k, int):
            raise ValueError("k must be an integer between 0 and n.")
        coeff = math.comb(self.n, k)
        return coeff * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k: int) -> float:
        if k < 0:
            return 0.0
        elif k >= self.n:
            return 1.0
        else:
            return sum(self.pmf(i) for i in range(0, k + 1))

    def expected_value(self) -> float:
        return self.n * self.p

    def variance(self) -> float:
        return self.n * self.p * (1 - self.p)

    def std_dev(self) -> float:
        return math.sqrt(self.variance())

# Geometric Distribution: Number of Bernoulli trials required for 1st success.
class GeometricDistribution(DiscreteDistribution):
    def __init__(self, p: float):
        if not (0.0 <= p <= 1.0):
            raise ValueError("Probability p must be in between 0 and 1.")
        self.p = p

    def pmf(self, k: int) -> float:
        if k < 1 or not isinstance(k, int):
            raise ValueError("k must be a positive integer.")
        return ((1.0 - self.p) ** (k - 1)) * self.p

    def cdf(self, k: int) -> float:
        if k < 1 or not isinstance(k, int):
            raise ValueError("k must be a positive integer.")
        return 1 - ((1.0 - self.p) ** math.floor(k))

    def expected_value(self) -> float:
        return 1.0 / self.p

    def variance(self) -> float:
        return (1.0 - self.p) / (self.p ** 2)

    def std_dev(self) -> float:
        return math.sqrt(self.variance())

# Poisson Distribution: Number of events in a fixed time interval.
class PoissonDistribution(DiscreteDistribution):
    def __init__(self, rate_lambda: float):
        if rate_lambda <= 0:
            raise ValueError("Lambda must be positive.")
        self.rate_lambda = rate_lambda

    def pmf(self, k: int) -> float:
        if k < 0 or not isinstance(k, int):
            raise ValueError("k must be a non-negative integer.")
        return (math.exp(-self.rate_lambda) * (self.rate_lambda ** k)) / math.factorial(k)

    def cdf(self, k: int) -> float:
        if k < 0 or not isinstance(k, int):
            raise ValueError("k must be a non-negative integer.")
        return sum(self.pmf(i) for i in range(0, k + 1))

    def expected_value(self) -> float:
        return self.rate_lambda

    def variance(self) -> float:
        return self.rate_lambda  

    def std_dev(self) -> float:
        return math.sqrt(self.variance()) 

# Continuous Distributions
# Continuous Uniform Distributions over interval [a, b].
class ContinuousUniformDistribution(ContinuousDistribution):
    def __init__(self, a: float, b: float):
        if a >= b:
            raise ValueError("Lower bound a must be less than upper bound b.")
        self.a = a
        self.b = b

    def pdf(self, x: float) -> float:
        if self.a <= x <= self.b:
            return 1.0 / (self.b - self.a)
        return 0.0

    def cdf(self, x: float) -> float:
        if x < self.a:
            return 0.0
        elif x > self.b:
            return 1.0
        else:
            return (x - self.a) / (self.b - self.a)

    def expected_value(self) -> float:
        return (self.a + self.b) / 2.0

    def variance(self) -> float:
        return ((self.b - self.a) ** 2) / 12.0

    def std_dev(self) -> float:
        return math.sqrt(self.variance())

# Exponential Distribution: Time between events in a Poisson point process.
class ExponentialDistribution(ContinuousDistribution):
    def __init__(self, rate_lambda: float):
        if rate_lambda < 0:
            raise ValueError("Rate parameter lambda must be non-negative.")
        self.rate_lambda = rate_lambda

    def pdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return self.rate_lambda * math.exp(-self.rate_lambda * x)

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return 1 - math.exp(-self.rate_lambda * x)

    def expected_value(self) -> float:
        return 1.0 / self.rate_lambda

    def variance(self) -> float:
        return 1.0 / (self.rate_lambda ** 2)

    def std_dev(self) -> float:
        return math.sqrt(self.variance())

# Gaussian (Normal) Distribution N(μ, σ²).
class GaussianDistribution(ContinuousDistribution):
    def __init__(self, mean: float, std_dev: float):
        if std_dev <= 0:
            raise ValueError("Standard deviation std_dev must be positive.")
        self.mu = mean
        self.sigma = std_dev

    def pdf(self, x: float) -> float:
        coeff = 1.0 / (self.sigma * math.sqrt(2 * math.pi))
        exponent = -((x - self.mu) ** 2) / (2 * self.sigma ** 2)
        return coeff * math.exp(exponent)

    def cdf(self, x: float) -> float:
        z = (x - self.mu) / (self.sigma * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def expected_value(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.sigma ** 2

    def std_dev(self) -> float:
        return math.sqrt(self.variance())

# Gamma Distribution parametrized by shape (α) and rate (β).
class GammaDistribution(ContinuousDistribution):
    def __init__(self, shape_alpha: float, rate_beta: float):
        if shape_alpha <= 0 or rate_beta <= 0:
            raise ValueError("Shape alpha and rate beta must be positive.")
        self.alpha = shape_alpha
        self.beta = rate_beta

    def pdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        coeff = (self.beta ** self.alpha) / math.gamma(self.alpha)
        return coeff * (x ** (self.alpha - 1)) * math.exp(-self.beta * x)

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        # Using numerical integration for CDF
        return self._integrate_simpson(self.pdf, 0, x)

    def expected_value(self) -> float:
        return self.alpha / self.beta

    def variance(self) -> float:
        return self.alpha / (self.beta ** 2)   

    def std_dev(self) -> float:
        return math.sqrt(self.variance()) 

# Beta Distribution over domain x ∈ (0, 1) with shape parameters α and β.
class BetaDistribution(ContinuousDistribution):
    def __init__(self, alpha: float, beta: float):
        if alpha <= 0 or beta <= 0:
            raise ValueError("Shape parameters alpha and beta must be positive.")
        self.alpha = alpha
        self.beta = beta

    def pdf(self, x: float) -> float:
        if x < 0 or x > 1:
            return 0.0
        coeff = math.gamma(self.alpha + self.beta) / (math.gamma(self.alpha) * math.gamma(self.beta))
        return coeff * (x ** (self.alpha - 1)) * ((1 - x) ** (self.beta - 1))

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        elif x > 1:
            return 1.0
        # Using numerical integration for CDF
        return self._integrate_simpson(self.pdf, 0, x)

    def expected_value(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        numerator = self.alpha * self.beta
        denominator = (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        return numerator / denominator   

    def std_dev(self) -> float:
        return math.sqrt(self.variance()) 
