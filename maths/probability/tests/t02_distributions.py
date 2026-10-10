import math
import pytest

from maths.probability.core.m00_probability_foundations import (
    SampleSpace,
    Event,
    ProbabilityMeasure,
    ConditionalProbability,
    RandomVariable
)

from maths.probability.core.m02_distributions import (
    BernoulliDistribution,
    BinomialDistribution,
    GeometricDistribution,
    PoissonDistribution,
    ContinuousUniformDistribution,
    ExponentialDistribution,
    GaussianDistribution,
    GammaDistribution,
    BetaDistribution
)

def test_discrete_distributions():
    bernoulli = BernoulliDistribution(0.25)
    assert math.isclose(bernoulli.pmf(1), 0.25)
    assert math.isclose(bernoulli.pmf(0), 0.75)
    assert math.isclose(bernoulli.cdf(0), 0.75)
    assert math.isclose(bernoulli.expected_value(), 0.25)
    assert math.isclose(bernoulli.variance(), 0.1875)

    binomial = BinomialDistribution(4, 0.5)
    assert math.isclose(binomial.pmf(2), 0.375)
    assert math.isclose(binomial.cdf(2), 0.6875)
    assert math.isclose(binomial.expected_value(), 2)
    assert math.isclose(binomial.variance(), 1)

    geometric = GeometricDistribution(0.5)
    assert math.isclose(geometric.pmf(3), 0.125)
    assert math.isclose(geometric.cdf(3), 0.875)

    poisson = PoissonDistribution(2)
    assert math.isclose(poisson.pmf(0), math.exp(-2))
    assert math.isclose(poisson.expected_value(), 2)
    assert math.isclose(poisson.variance(), 2)


def test_continuous_distributions():
    uniform = ContinuousUniformDistribution(0, 2)
    assert math.isclose(uniform.pdf(1), 0.5)
    assert math.isclose(uniform.cdf(1), 0.5)
    assert math.isclose(uniform.expected_value(), 1)
    assert math.isclose(uniform.variance(), 1 / 3)

    exponential = ExponentialDistribution(2)
    assert math.isclose(exponential.pdf(0), 2)
    assert math.isclose(exponential.cdf(1), 1 - math.exp(-2))
    assert math.isclose(exponential.expected_value(), 0.5)
    assert math.isclose(exponential.variance(), 0.25)

    gaussian = GaussianDistribution(0, 1)
    assert math.isclose(gaussian.pdf(0), 1 / math.sqrt(2 * math.pi))
    assert math.isclose(gaussian.cdf(0), 0.5)

    gamma = GammaDistribution(2, 2)
    assert math.isclose(gamma.expected_value(), 1)
    assert math.isclose(gamma.variance(), 0.5)
    assert math.isclose(gamma.cdf(1), 1 - 3 * math.exp(-2), rel_tol=1e-3)

    beta = BetaDistribution(2, 2)
    assert math.isclose(beta.pdf(0.5), 1.5)
    assert math.isclose(beta.cdf(0.5), 0.5, rel_tol=1e-3)


def test_probability_validation():
    omega = SampleSpace({1, 2})

    with pytest.raises(ValueError):
        ProbabilityMeasure(omega, {1: 0.5})

    with pytest.raises(ValueError):
        ProbabilityMeasure(omega, {1: 0.3, 2: 0.3})

    with pytest.raises(ValueError):
        ConditionalProbability(ProbabilityMeasure(omega)).compute(
            Event("A", {1}, omega),
            Event("B", set(), omega),
        )
