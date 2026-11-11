"""
Asymptotic / convergence-order / limit theorem / complexity tests for maths.probability.
"""

import math

import numpy as np
import pytest

from maths.probability.core.m00_probability_foundations import Event, ProbabilityMeasure, SampleSpace
from maths.probability.core.m02_distributions import (
    BinomialDistribution, ContinuousDistribution, GaussianDistribution, PoissonDistribution)
from maths.test_helpers import best_time, growth_exponent, time_limit

simpson = ContinuousDistribution._integrate_simpson

# Simpson convergence order
@pytest.mark.scaling
class TestSimpsonRule:
    def test_exact_for_cubics(self):
        f = lambda x: 2 * x ** 3 - x ** 2 + 4 * x - 1
        assert simpson(f, -1.0, 2.0, 2) == pytest.approx(
            (2 / 4) * (16 - 1) - (1 / 3) * (8 + 1) + 2 * (4 - 1) - 3.0, rel=1e-13)

    def test_fourth_order_convergence_on_smooth_integrand(self):
        exact = math.e - 1
        errs = [abs(simpson(math.exp, 0.0, 1.0, n) - exact) for n in (4, 8, 16, 32, 64)]
        orders = [math.log2(a / b) for a, b in zip(errs, errs[1:])]
        assert all(3.7 <= o <= 4.3 for o in orders), f"observed orders {orders}"

    def test_odd_n_is_promoted_and_reversed_bounds_flip_sign(self):
        assert simpson(math.sin, 0.0, 1.0, 7) == pytest.approx(simpson(math.sin, 0.0, 1.0, 8), rel=1e-15)
        assert simpson(math.sin, 1.0, 0.0, 10) == pytest.approx(-simpson(math.sin, 0.0, 1.0, 10), rel=1e-15)
        assert simpson(math.sin, 0.5, 0.5, 10) == 0.0

    def test_integrable_endpoint_singularity_degrades_order(self):
        errs = [abs(simpson(math.sqrt, 0.0, 1.0, n) - 2 / 3) for n in (16, 32, 64, 128)]
        orders = [math.log2(a / b) for a, b in zip(errs, errs[1:])]
        assert all(o < 2.0 for o in orders)

    def test_fixed_panel_count_is_not_adaptive_to_interval_length(self):
        f = lambda x: math.exp(-x)
        for b in (10, 100, 1000, 10_000):
            assert simpson(f, 0.0, float(b), 1000) == pytest.approx(1 - math.exp(-b), abs=1e-6)

# limit theorems / asymptotic agreement
@pytest.mark.reference
class TestAsymptoticAgreement:
    @pytest.mark.parametrize("n", [50, 200, 800])
    def test_binomial_converges_to_normal_clt_with_continuity_correction(self, n):
        p = 0.3
        d, g = BinomialDistribution(n, p), GaussianDistribution(n * p, math.sqrt(n * p * (1 - p)))
        worst = max(abs(d.cdf(k) - g.cdf(k + 0.5)) for k in range(0, n, max(1, n // 40)))
        assert worst <= 0.5 / math.sqrt(n * p * (1 - p)) * 0.4          # Berry-Esseen order O(1/sqrt(n))

    def test_clt_error_shrinks_like_inverse_sqrt_n(self):
        def worst(n):
            d, g = BinomialDistribution(n, 0.3), GaussianDistribution(0.3 * n, math.sqrt(0.21 * n))
            return max(abs(d.cdf(k) - g.cdf(k + 0.5)) for k in range(int(0.3 * n) - 15, int(0.3 * n) + 16))
        ns = [100, 300, 900]            # stay below the n~1030 comb() overflow so the slope is what is measured
        slope = growth_exponent(ns, [worst(n) for n in ns])
        assert -1.3 <= slope <= -0.4, f"error ~ n^{slope:.2f}"

    @pytest.mark.parametrize("n", [100, 400, 1600])
    def test_le_cam_binomial_to_poisson(self, n):
        lam = 2.0
        b, po = BinomialDistribution(n, lam / n), PoissonDistribution(lam)
        tv_bound = n * (lam / n) ** 2                                   # Le Cam: d_TV <= n p^2
        sup = max(abs(b.pmf(k) - po.pmf(k)) for k in range(0, 25))
        assert sup <= tv_bound

    def test_poisson_approximation_error_decays_like_one_over_n(self):
        lam, ns = 2.0, [100, 400, 1600]
        errs = [max(abs(BinomialDistribution(n, lam / n).pmf(k) - PoissonDistribution(lam).pmf(k)) for k in range(25)) for n in ns]
        slope = growth_exponent(ns, errs)
        assert -1.2 <= slope <= -0.8, f"error ~ n^{slope:.2f}"

    @pytest.mark.parametrize("lam", [25, 64, 100])
    def test_poisson_converges_to_normal(self, lam):
        d, g = PoissonDistribution(lam), GaussianDistribution(lam, math.sqrt(lam))
        hi = min(int(lam + 5 * math.sqrt(lam)), 150)
        assert max(abs(d.cdf(k) - g.cdf(k + 0.5)) for k in range(int(lam - 4 * math.sqrt(lam)), hi)) <= 0.5 / math.sqrt(lam) * 0.3

    def test_law_of_large_numbers_by_exact_enumeration(self):
        p, eps = 0.4, 0.05
        def tail(n):
            d = BinomialDistribution(n, p)
            return sum(d.pmf(k) for k in range(n + 1) if abs(k / n - p) > eps)
        tails = [tail(n) for n in (50, 100, 200, 400, 800)]
        assert all(a > b for a, b in zip(tails, tails[1:]))
        assert all(t <= 2 * math.exp(-2 * n * eps ** 2) + 1e-12 for t, n in zip(tails, (50, 100, 200, 400, 800)))

# computational complexity
@pytest.mark.scaling
class TestComplexity:
    def test_binomial_cdf_cost_grows_linearly_in_n(self):
        def t(n):
            d = BinomialDistribution(n, 0.5)
            return best_time(lambda: d.cdf(n - 1), 2)
        ns = [150, 300, 600, 1000]
        assert growth_exponent(ns, [t(n) for n in ns]) <= 1.4

    @pytest.mark.parametrize("k", [150, 400, 1000])
    def test_poisson_cdf_is_computable_for_large_k(self, k):
        with time_limit(5):
            assert PoissonDistribution(float(k) / 2).cdf(k) == pytest.approx(1.0, abs=1e-3)

    def test_gamma_cdf_cost_is_constant_in_x(self):
        from maths.probability.core.m02_distributions import GammaDistribution
        d = GammaDistribution(2, 1)
        t1, t2 = best_time(lambda: d.cdf(1.0)), best_time(lambda: d.cdf(1e4))
        assert t2 <= 3 * t1 + 1e-3          

    def test_event_set_operations_are_linear(self):
        def t(n):
            S = SampleSpace(range(n))
            A, B = Event("a", range(0, n, 2), S), Event("b", range(0, n, 3), S)
            return best_time(lambda: (A | B, A & B, ~A, A - B), 3)
        ns = [50_000, 100_000, 200_000, 400_000]
        assert 0.7 <= growth_exponent(ns, [t(n) for n in ns]) <= 1.4

    def test_measure_over_a_million_outcomes(self):
        n = 10**6
        with time_limit(20):
            S = SampleSpace(range(n))
            P = ProbabilityMeasure(S)
            assert P(Event("half", range(n // 2), S)) == pytest.approx(0.5, abs=1e-10)
