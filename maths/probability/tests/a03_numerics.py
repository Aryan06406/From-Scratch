"""
Numerical-precision / reference-implementation / stress tests for maths.probability.
Oracles: scipy.stats / scipy.special (skipped if scipy is absent), math.erfc / lgamma, exact Fractions.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from maths.probability.core.m00_probability_foundations import Event, ProbabilityMeasure, SampleSpace
from maths.probability.core.m01_bayes_theorem import BayesTheorem
from maths.probability.core.m02_distributions import (
    BetaDistribution, BinomialDistribution, ExponentialDistribution, GammaDistribution,
    GaussianDistribution, GeometricDistribution, PoissonDistribution)

stats = pytest.importorskip("scipy.stats")
special = pytest.importorskip("scipy.special")

def rel(a, b):
    return abs(a - b) / abs(b) if b else abs(a)

# Gaussian
@pytest.mark.reference
class TestGaussianAgainstScipy:
    @pytest.mark.parametrize("z", [-37, -20, -10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 5, 8, 10])
    def test_cdf_relative_accuracy_including_both_tails(self, z):
        got, ref = GaussianDistribution(0, 1).cdf(z), stats.norm.cdf(z)
        assert rel(got, ref) <= 1e-12, f"cdf({z}) = {got!r}, scipy = {ref!r}"

    @pytest.mark.parametrize("mu,sigma,x", [(0, 1, 0.3), (5, 0.01, 5.02), (-1e6, 1e3, -1.0045e6), (0, 1e-5, 3e-5), (100, 1e4, -4e4)])
    def test_cdf_pdf_with_odd_scales(self, mu, sigma, x):
        d = GaussianDistribution(mu, sigma)
        assert d.cdf(x) == pytest.approx(stats.norm.cdf(x, mu, sigma), rel=1e-11, abs=1e-300)
        assert d.pdf(x) == pytest.approx(stats.norm.pdf(x, mu, sigma), rel=1e-11, abs=1e-300)

    @pytest.mark.parametrize("z", [30, 37, 38.5])
    def test_pdf_deep_tail(self, z):
        assert GaussianDistribution(0, 1).pdf(z) == pytest.approx(stats.norm.pdf(z), rel=1e-12, abs=1e-320)

    def test_cdf_sum_symmetry_holds_to_machine_precision(self):
        d = GaussianDistribution(0, 1)
        for z in (0.1, 1, 2.5, 4):
            assert d.cdf(z) + d.cdf(-z) == pytest.approx(1.0, abs=2e-16)


# Exponential / Geometric
@pytest.mark.numeric
class TestSmallProbabilityPrecision:
    @pytest.mark.parametrize("x", [1e-3, 1e-8, 1e-12, 1e-16, 1e-20, 1e-100])
    def test_exponential_cdf_small_argument(self, x):
        assert ExponentialDistribution(1.0).cdf(x) == pytest.approx(-math.expm1(-x), rel=1e-12)

    @pytest.mark.parametrize("p,k", [(1e-3, 5), (1e-10, 3), (1e-16, 10), (1e-17, 10), (1e-30, 7)])
    def test_geometric_cdf_tiny_p(self, p, k):
        ref = -math.expm1(k * math.log1p(-p))
        assert GeometricDistribution(p).cdf(k) == pytest.approx(ref, rel=1e-9)

    @pytest.mark.parametrize("lam,x", [(0.5, 10), (3, 2), (1e-6, 1e6), (1e3, 0.01)])
    def test_exponential_matches_scipy(self, lam, x):
        d, r = ExponentialDistribution(lam), stats.expon(scale=1 / lam)
        assert d.cdf(x) == pytest.approx(r.cdf(x), rel=1e-12)
        assert d.pdf(x) == pytest.approx(r.pdf(x), rel=1e-12)

# Binomial / Poisson
@pytest.mark.reference
class TestDiscreteAgainstScipy:
    @pytest.mark.parametrize("n,p", [(10, 0.3), (100, 0.5), (500, 0.01), (1000, 0.5), (1000, 0.99)])
    def test_binomial_pmf_cdf_moderate_n(self, n, p):
        d, r = BinomialDistribution(n, p), stats.binom(n, p)
        ks = np.unique(np.linspace(0, n, 25).astype(int))
        for k in ks:
            k = int(k)
            assert d.pmf(k) == pytest.approx(r.pmf(k), rel=1e-9, abs=1e-300)
        for k in ks[:-1]:
            assert d.cdf(int(k)) == pytest.approx(r.cdf(k), rel=1e-9, abs=1e-12)

    @pytest.mark.parametrize("n", [1030, 1100, 2000, 10_000])
    def test_binomial_pmf_large_n(self, n):
        d = BinomialDistribution(n, 0.5)
        assert d.pmf(n // 2) == pytest.approx(stats.binom.pmf(n // 2, n, 0.5), rel=1e-9)

    @pytest.mark.parametrize("lam", [0.01, 1, 10, 50, 100])
    def test_poisson_pmf_cdf_moderate_lambda(self, lam):
        d, r = PoissonDistribution(lam), stats.poisson(lam)
        for k in range(0, int(lam + 6 * math.sqrt(lam) + 5)):
            if k > 150:
                break
            assert d.pmf(k) == pytest.approx(r.pmf(k), rel=1e-9, abs=1e-300)
        k = min(int(lam), 150)
        assert d.cdf(k) == pytest.approx(r.cdf(k), rel=1e-9)

    @pytest.mark.parametrize("lam,k", [(200, 200), (500, 500), (1000, 1000), (800, 750)])
    def test_poisson_pmf_large_lambda(self, lam, k):
        assert PoissonDistribution(lam).pmf(k) == pytest.approx(stats.poisson.pmf(k, lam), rel=1e-9)

    @pytest.mark.parametrize("lam,k", [(1.0, 171), (1.0, 200), (5.0, 500), (2.0, 1000)])
    def test_poisson_cdf_far_in_the_tail_is_one(self, lam, k):
        assert PoissonDistribution(lam).cdf(k) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("p,k", [(0.5, 1), (0.01, 200), (0.01, 5000), (0.9, 30)])
    def test_geometric_matches_scipy(self, p, k):
        d, r = GeometricDistribution(p), stats.geom(p)
        assert d.pmf(k) == pytest.approx(r.pmf(k), rel=1e-9, abs=1e-300)
        assert d.cdf(k) == pytest.approx(r.cdf(k), rel=1e-9)

# Gamma / Beta (Simpson based CDF)
@pytest.mark.reference
class TestGammaBetaAgainstScipy:
    @pytest.mark.parametrize("alpha,beta", [(1, 1), (2, 1), (2.5, 0.5), (7, 3), (20, 1)])
    def test_gamma_pdf_matches_scipy(self, alpha, beta):
        d = GammaDistribution(alpha, beta)
        for x in (0.1, alpha / beta, 3 * alpha / beta):
            assert d.pdf(x) == pytest.approx(stats.gamma.pdf(x, alpha, scale=1 / beta), rel=1e-10)

    @pytest.mark.parametrize("x", [1, 5, 10, 30, 100, 300, 1000, 10_000])
    def test_gamma_cdf_accuracy_does_not_degrade_with_x(self, x):
        got, ref = GammaDistribution(2, 1).cdf(x), special.gammainc(2, x)
        assert abs(got - ref) <= 1e-9, f"cdf({x}) = {got}, scipy = {ref}"

    @pytest.mark.parametrize("alpha", [1.5, 2.5, 7.3])
    def test_gamma_cdf_non_integer_shape_accuracy(self, alpha):
        for x in (0.5, 2.0, 10.0):
            assert GammaDistribution(alpha, 1).cdf(x) == pytest.approx(special.gammainc(alpha, x), abs=1e-9)

    @pytest.mark.parametrize("alpha", [150, 180, 200, 500])
    def test_gamma_large_shape_overflows_in_gamma_function(self, alpha):
        assert GammaDistribution(alpha, 1).pdf(alpha) == pytest.approx(stats.gamma.pdf(alpha, alpha), rel=1e-9)

    @pytest.mark.parametrize("a,b", [(2, 3), (5, 5), (10, 2)])
    def test_beta_pdf_cdf_integer_shapes(self, a, b):
        d = BetaDistribution(a, b)
        for x in (0.05, 0.3, 0.5, 0.9):
            assert d.pdf(x) == pytest.approx(stats.beta.pdf(x, a, b), rel=1e-10)
            assert d.cdf(x) == pytest.approx(stats.beta.cdf(x, a, b), abs=1e-10)

    @pytest.mark.parametrize("a,b", [(1.5, 2.5), (2.2, 1.7), (3.5, 1.2)])
    def test_beta_cdf_non_integer_shapes(self, a, b):
        d = BetaDistribution(a, b)
        for x in (0.2, 0.5, 0.95, 1.0):
            assert d.cdf(x) == pytest.approx(stats.beta.cdf(x, a, b), abs=1e-9)

    @pytest.mark.parametrize("a", [100, 172, 300])
    def test_beta_large_shapes_overflow_in_gamma_function(self, a):
        assert BetaDistribution(a, a).pdf(0.5) == pytest.approx(stats.beta.pdf(0.5, a, a), rel=1e-9)

    def test_beta_sharply_peaked_cdf(self):
        assert BetaDistribution(60, 60).cdf(0.55) == pytest.approx(stats.beta.cdf(0.55, 60, 60), abs=1e-9)

# Foundations / Bayes numerics
@pytest.mark.numeric
class TestMeasureAndBayesNumerics:
    @pytest.mark.parametrize("n", [3, 7, 10, 1000, 10**5, 10**6])
    def test_uniform_measure_total_mass(self, n):
        S = SampleSpace(range(n))
        P = ProbabilityMeasure(S)
        assert abs(P(Event("Ω", range(n), S)) - 1.0) <= 1e-12

    @pytest.mark.parametrize("c", [1e-300, 1e-310])
    def test_posterior_invariant_to_likelihood_scale_down_to_underflow(self, c):
        pri, lik = {"a": 0.25, "b": 0.75}, {"a": 0.8, "b": 0.2}
        base = BayesTheorem.compute_from_partition(pri, lik, "a")
        got = BayesTheorem.compute_from_partition(pri, {k: v * c for k, v in lik.items()}, "a")
        assert got == pytest.approx(base, rel=1e-6)

    @pytest.mark.parametrize("prior,tpr,fpr", [(1e-9, 0.999, 1e-3), (1e-12, 1.0, 1e-6), (0.5, 1e-300, 1e-300), (1 - 1e-12, 0.9, 0.1)])
    def test_extreme_priors_against_exact_fractions(self, prior, tpr, fpr):
        pri, lik = {"H": prior, "N": 1 - prior}, {"H": tpr, "N": fpr}
        F = {k: Fraction(v) for k, v in {**{"pH": prior, "pN": 1 - prior}, "tH": tpr, "tN": fpr}.items()}
        exact = (F["tH"] * F["pH"]) / (F["tH"] * F["pH"] + F["tN"] * F["pN"])
        assert BayesTheorem.compute_from_partition(pri, lik, "H") == pytest.approx(float(exact), rel=1e-12)
