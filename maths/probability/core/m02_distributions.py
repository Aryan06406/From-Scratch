from __future__ import annotations

import math
from typing import Callable
from abc import ABC, abstractmethod
from numbers import Integral, Real

from maths.probability.core.m00_probability_foundations import ProbabilityMeasure, SampleSpace, Event, RandomVariable

_LOG_MIN_SUBNORMAL = math.log(float.fromhex("0x0.0000000000001p-1022"))
_LOG_MAX = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)

def _finite_real(value: object, name: str) -> float:
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result

def _integer_k(k: object, description: str) -> int:
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise ValueError(description)
    return int(k)

def _cdf_floor(k: object) -> int | float:
    """CDF arguments are real; return floor(k), preserving +/- infinity."""
    if not isinstance(k, Real):
        raise TypeError("CDF argument must be a real number.")
    value = float(k)
    if math.isnan(value):
        raise ValueError("CDF argument cannot be NaN.")
    if math.isinf(value):
        return value
    return math.floor(value)

def _exp_from_log(log_value: float) -> float:
    if log_value < _LOG_MIN_SUBNORMAL:
        return 0.0
    if log_value > _LOG_MAX:
        return math.inf
    return math.exp(log_value)

def _regularized_gamma_p(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x), using series/continued fraction."""
    if a <= 0.0 or math.isnan(a) or math.isnan(x):
        raise ValueError("The regularized gamma function requires a > 0 and non-NaN x.")
    if x <= 0.0:
        return 0.0
    if math.isinf(x):
        return 1.0

    gln = math.lgamma(a)
    eps = 2.0e-14
    max_iter = 10000
    fpmin = 1.0e-300

    if x < a + 1.0:
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(max_iter):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) <= abs(total) * eps:
                break
        result = total * _exp_from_log(-x + a * math.log(x) - gln)
        return min(1.0, max(0.0, result))

    # Compute Q(a,x) with a modified Lentz continued fraction, then P = 1-Q.
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / max(abs(b), fpmin)
    if b < 0.0:
        d = -d
    h = d
    for i in range(1, max_iter + 1):
        an = -float(i) * (float(i) - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= eps:
            break
    q = _exp_from_log(-x + a * math.log(x) - gln) * h
    return min(1.0, max(0.0, 1.0 - q))

def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iter = 10000
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= eps:
            break
    return h

def _regularized_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b), using a continued fraction."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("Beta shape parameters must be positive.")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    bt = _exp_from_log(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        result = bt * _beta_continued_fraction(a, b, x) / a
    else:
        result = 1.0 - bt * _beta_continued_fraction(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, result))


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
        """Composite Simpson integration, refining broad intervals to keep h <= 0.1."""

        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
            raise ValueError("n must be a positive integer.")
        if a == b:
            return 0.0
        if b < a:
            return -ContinuousDistribution._integrate_simpson(f, b, a, n)
        
        n = int(n)
        # On very wide intervals, a fixed n can make h enormous and miss the
        # region where the integrand has mass. Refine only those broad spans;
        # preserving n on ordinary intervals retains the expected O(h^4) study.
        
        span = b - a
        
        if span > 100.0:
            n = max(n, int(math.ceil(span / 0.1)))
        if n % 2:
            n += 1
        h = (b - a) / n
        
        total = f(a) + f(b)
        total += 4.0 * math.fsum(f(a + i * h) for i in range(1, n, 2))
        total += 2.0 * math.fsum(f(a + i * h) for i in range(2, n - 1, 2))
        
        return total * h / 3.0

# Discrete Distributions
# Bernoulli Distribution: Single trial with success probability p.
class BernoulliDistribution(DiscreteDistribution):
    def __init__(self, p: float):
        self.p = _finite_real(p, "p")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("Probability p must lie in [0, 1].")

    def pmf(self, k: int) -> float:
        k = _integer_k(k, "k must be 0 or 1 for Bernoulli distribution.")
        if k not in (0, 1):
            raise ValueError("k must be 0 or 1 for Bernoulli distribution.")
        return self.p if k == 1 else 1.0 - self.p

    def cdf(self, k: float) -> float:
        k = _cdf_floor(k)
        if k < 0:
            return 0.0
        if k < 1:
            return 1.0 - self.p
        return 1.0

    def expected_value(self) -> float:
        return self.p

    def variance(self) -> float:
        return self.p * (1.0 - self.p) 

    def std_dev(self) -> float:
        return math.sqrt(self.variance())       

# Binomial Distribution: Number of successes in n independent Bernoulli trials.
class BinomialDistribution(DiscreteDistribution):
    def __init__(self, n: int, p: float):
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
            raise ValueError("Number of trials n must be a positive integer.")
        self.n = int(n)
        self.p = _finite_real(p, "p")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("Probability p must lie in [0, 1].")

    def _log_pmf(self, k: int) -> float:
        if self.p == 0.0:
            return 0.0 if k == 0 else -math.inf
        if self.p == 1.0:
            return 0.0 if k == self.n else -math.inf
        return (
            math.lgamma(self.n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(self.n - k + 1.0)
            + k * math.log(self.p) + (self.n - k) * math.log1p(-self.p)
        )

    def pmf(self, k: int) -> float:
        k = _integer_k(k, "k must be an integer between 0 and n.")
        if k < 0 or k > self.n:
            raise ValueError("k must be an integer between 0 and n.")
        return _exp_from_log(self._log_pmf(k))

    def cdf(self, k: float) -> float:
        k = _cdf_floor(k)

        if k < 0:
            return 0.0
        if k >= self.n:
            return 1.0
        
        k = int(k)
        if self.p == 0.0:
            return 1.0
        if self.p == 1.0:
            return 0.0

        # Sum the shorter tail, starting at its largest/nearest probability.
        # This avoids both O(n^2) repeated combinations and underflow at P(X=0).
        if k <= self.n * self.p:
            term = self.pmf(k)
            terms = [term]
            for i in range(k, 0, -1):
                term *= (i / (self.n - i + 1.0)) * ((1.0 - self.p) / self.p)
                terms.append(term)
            result = math.fsum(terms)
        
        else:
            first = k + 1
            term = self.pmf(first)
            terms = [term]
            for i in range(first, self.n):
                term *= ((self.n - i) / (i + 1.0)) * (self.p / (1.0 - self.p))
                terms.append(term)
            result = 1.0 - math.fsum(terms)
        
        return min(1.0, max(0.0, result))

    def expected_value(self) -> float:
        return self.n * self.p

    def variance(self) -> float:
        return self.n * self.p * (1.0 - self.p)

    def std_dev(self) -> float:
        return math.sqrt(self.variance())

# Geometric Distribution: Number of Bernoulli trials required for 1st success.
class GeometricDistribution(DiscreteDistribution):
    def __init__(self, p: float):
        self.p = _finite_real(p, "p")
        if not 0.0 < self.p <= 1.0:
            raise ValueError("Geometric probability p must lie in (0, 1].")

    def pmf(self, k: int) -> float:
        k = _integer_k(k, "k must be a positive integer.")
        if k < 1:
            raise ValueError("k must be a positive integer.")
        if self.p == 1.0:
            return 1.0 if k == 1 else 0.0
        return _exp_from_log((k - 1) * math.log1p(-self.p) + math.log(self.p))

    def cdf(self, k: float) -> float:
        k = _cdf_floor(k)
        if k < 1:
            return 0.0
        if math.isinf(k) or self.p == 1.0:
            return 1.0
        return min(1.0, max(0.0, -math.expm1(int(k) * math.log1p(-self.p))))

    def expected_value(self) -> float:
        return 1.0 / self.p

    def variance(self) -> float:
        return (1.0 - self.p) / (self.p * self.p)

    def std_dev(self) -> float:
        return math.sqrt(self.variance())

# Poisson Distribution: Number of events in a fixed time interval.
class PoissonDistribution(DiscreteDistribution):
    def __init__(self, rate_lambda: float):
        self.rate_lambda = _finite_real(rate_lambda, "rate_lambda")
        if self.rate_lambda <= 0.0:
            raise ValueError("Lambda must be positive.")

    def _log_pmf(self, k: int) -> float:
        return (-self.rate_lambda + k * math.log(self.rate_lambda) - math.lgamma(k + 1.0))

    def pmf(self, k: int) -> float:
        k = _integer_k(k, "k must be a non-negative integer.")
        if k < 0:
            raise ValueError("k must be a non-negative integer.")
        return _exp_from_log(self._log_pmf(k))

    def cdf(self, k: float) -> float:
        k = _cdf_floor(k)
        if k < 0:
            return 0.0
        if math.isinf(k):
            return 1.0
        k = int(k)
        lam = self.rate_lambda

        # For k below lambda, add the lower tail downwards from P(X=k).
        if k < lam:
            term = self.pmf(k)
            terms = [term]
            for i in range(k, 0, -1):
                term *= i / lam
                terms.append(term)
            return min(1.0, max(0.0, math.fsum(terms)))

        # Otherwise compute the upper tail P(X>k), then take its complement.
        # The terms decrease monotonically in this region.
        term = self.pmf(k + 1)
        if term == 0.0:
            return 1.0
        
        terms = [term]
        running_total = term
        i = k + 1
        
        while term > 0.0:
            i += 1
            term *= lam / i
            if term == 0.0:
                break
            terms.append(term)
            running_total += term
            if term <= running_total * 1e-16:
                break
       
        upper_tail = math.fsum(terms)
        return min(1.0, max(0.0, 1.0 - upper_tail))

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
        self.a = _finite_real(a, "a")
        self.b = _finite_real(b, "b")
        if self.a >= self.b:
            raise ValueError("Lower bound a must be less than upper bound b.")

    def pdf(self, x: float) -> float:
        return 1.0 / (self.b - self.a) if self.a <= x <= self.b else 0.0

    def cdf(self, x: float) -> float:
        if x <= self.a:
            return 0.0
        if x >= self.b:
            return 1.0
        return (x - self.a) / (self.b - self.a)

    def expected_value(self) -> float:
        return (self.a + self.b) / 2.0

    def variance(self) -> float:
        width = self.b - self.a
        return (width * width) / 12.0

    def std_dev(self) -> float:
        return math.sqrt(self.variance())

# Exponential Distribution: Time between events in a Poisson point process.
class ExponentialDistribution(ContinuousDistribution):
    def __init__(self, rate_lambda: float):
        self.rate_lambda = _finite_real(rate_lambda, "rate_lambda")
        if self.rate_lambda <= 0.0:
            raise ValueError("Rate parameter lambda must be positive.")

    def pdf(self, x: float) -> float:
        if x < 0.0:
            return 0.0
        return self.rate_lambda * math.exp(-self.rate_lambda * x)

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        return -math.expm1(-self.rate_lambda * x)

    def expected_value(self) -> float:
        return 1.0 / self.rate_lambda

    def variance(self) -> float:
        return 1.0 / (self.rate_lambda * self.rate_lambda)

    def std_dev(self) -> float:
        return math.sqrt(self.variance())

# Gaussian (Normal) Distribution N(μ, σ²).
class GaussianDistribution(ContinuousDistribution):
    def __init__(self, mean: float, std_dev: float):
        self.mu = _finite_real(mean, "mean")
        self.sigma = _finite_real(std_dev, "std_dev")
        if self.sigma <= 0.0:
            raise ValueError("Standard deviation std_dev must be positive.")

    def pdf(self, x: float) -> float:
        x = float(x)
        
        if math.isnan(x):
            return math.nan
        z = (x - self.mu) / self.sigma
        
        if not math.isfinite(z) or abs(z) > 39.0:
            return 0.0
        return math.exp(-0.5 * z * z) / (self.sigma * _SQRT_2PI)

    def cdf(self, x: float) -> float:
        x = float(x)
        if math.isnan(x):
            return math.nan
        z = (x - self.mu) / (self.sigma * _SQRT_2)
        return 0.5 * math.erfc(-z)

    def expected_value(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.sigma * self.sigma

    def std_dev(self) -> float:
        return math.sqrt(self.variance())

# Gamma Distribution parametrized by shape (α) and rate (β).
class GammaDistribution(ContinuousDistribution):
    def __init__(self, shape_alpha: float, rate_beta: float):
        self.alpha = _finite_real(shape_alpha, "shape_alpha")
        self.beta = _finite_real(rate_beta, "rate_beta")
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError("Shape alpha and rate beta must be positive.")

    def pdf(self, x: float) -> float:
        x = float(x)
        if x < 0.0:
            return 0.0

        if x == 0.0:
            if self.alpha < 1.0:
                return math.inf
            if self.alpha == 1.0:
                return self.beta
            return 0.0

        log_density = (
            self.alpha * math.log(self.beta) - math.lgamma(self.alpha)
            + (self.alpha - 1.0) * math.log(x) - self.beta * x
        )
        
        return _exp_from_log(log_density)

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        return _regularized_gamma_p(self.alpha, self.beta * x)

    def expected_value(self) -> float:
        return self.alpha / self.beta

    def variance(self) -> float:
        return self.alpha / (self.beta * self.beta)

    def std_dev(self) -> float:
        return math.sqrt(self.variance()) 

# Beta Distribution over domain x ∈ (0, 1) with shape parameters α and β.
class BetaDistribution(ContinuousDistribution):
    def __init__(self, alpha: float, beta: float):
        self.alpha = _finite_real(alpha, "alpha")
        self.beta = _finite_real(beta, "beta")
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError("Shape parameters alpha and beta must be positive.")

    def pdf(self, x: float) -> float:
        x = float(x)
        if x < 0.0 or x > 1.0:
            return 0.0
        
        if x == 0.0:
            if self.alpha < 1.0:
                return math.inf
            if self.alpha > 1.0:
                return 0.0
            return self.beta
        
        if x == 1.0:
            if self.beta < 1.0:
                return math.inf
            if self.beta > 1.0:
                return 0.0
            return self.alpha
        
        log_density = (
            math.lgamma(self.alpha + self.beta) - math.lgamma(self.alpha) - math.lgamma(self.beta)
            + (self.alpha - 1.0) * math.log(x) + (self.beta - 1.0) * math.log1p(-x)
        )
        
        return _exp_from_log(log_density)

    def cdf(self, x: float) -> float:
        x = float(x)
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        return _regularized_beta(self.alpha, self.beta, x)

    def expected_value(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1.0))


    def std_dev(self) -> float:
        return math.sqrt(self.variance()) 
