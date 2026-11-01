"""
Property-based + invariance/symmetry tests for maths.probability.
Random finite probability spaces, set-algebra laws, Kolmogorov axioms, Bayes identities,
and distribution identities (recurrences, additivity, location-scale, cross-family equalities).
Parameters are drawn from regions where each distribution is numerically sane, so a failure here
means the maths is wrong, not that float64 ran out of range (that is covered in test_prob_numerics).
"""
import math

import pytest
from hypothesis import assume, given, strategies as st

from maths.probability.core.m00_probability_foundations import (
    ConditionalProbability, Event, ProbabilityMeasure, SampleSpace, SetOperations)
from maths.probability.core.m01_bayes_theorem import BayesTheorem
from maths.probability.core.m02_distributions import (
    BernoulliDistribution, BetaDistribution, BinomialDistribution, ContinuousUniformDistribution,
    ExponentialDistribution, GammaDistribution, GaussianDistribution, GeometricDistribution,
    PoissonDistribution)

TOL = 1e-12


# ---------------------------------------------------------------- strategies
@st.composite
def spaces_and_measure(draw, max_n=8):
    n = draw(st.integers(1, max_n))
    S = SampleSpace(range(n))
    raw = draw(st.lists(st.integers(0, 20), min_size=n, max_size=n))
    assume(sum(raw) > 0)
    tot = sum(raw)
    return S, ProbabilityMeasure(S, {i: w / tot for i, w in enumerate(raw)})


def subsets(S):
    return st.sets(st.sampled_from(sorted(S.outcomes)))


@st.composite
def model_with_events(draw, k=3):
    S, P = draw(spaces_and_measure())
    evs = [Event(f"E{i}", draw(subsets(S)), S) for i in range(k)]
    return S, P, evs


def full(S):
    return Event("Ω", S.outcomes, S)


def empty(S):
    return Event("∅", [], S)


# ================================================================ event algebra
@pytest.mark.property
class TestEventAlgebra:
    @given(model_with_events())
    def test_boolean_algebra_laws(self, m):
        S, _, (A, B, C) = m
        assert A | B == B | A and A & B == B & A
        assert (A | B) | C == A | (B | C) and (A & B) & C == A & (B & C)
        assert A & (B | C) == (A & B) | (A & C) and A | (B & C) == (A | B) & (A | C)
        assert ~(A | B) == ~A & ~B and ~(A & B) == ~A | ~B                      # De Morgan
        assert ~~A == A and A | A == A and A & A == A
        assert A | (A & B) == A and A & (A | B) == A                           # absorption
        assert A - B == A & ~B and A | ~A == full(S) and A & ~A == empty(S)
        assert A | empty(S) == A and A & full(S) == A and A & empty(S) == empty(S)
        assert (A & B).is_subset(A) and A.is_subset(A | B) and (A | B).is_superset(B)
        assert A.is_disjoint(B) == (A & B).is_empty()
        assert A.is_subset(B) == ((A & B) == A) == ((A | B) == B)

    @given(model_with_events())
    def test_operators_match_set_operations_and_cardinality(self, m):
        S, _, (A, B, _) = m
        assert A | B == SetOperations.union(A, B) and A & B == SetOperations.intersection(A, B)
        assert (A | B).cardinality() == A.cardinality() + B.cardinality() - (A & B).cardinality()
        assert A.copy() == A and A.copy() is not A
        for w in S.outcomes:
            assert (w in A) != (w in ~A)


# ================================================================ Kolmogorov axioms
@pytest.mark.property
class TestProbabilityAxioms:
    @given(model_with_events())
    def test_axioms_and_derived_rules(self, m):
        S, P, (A, B, C) = m
        assert P(empty(S)) == 0 and abs(P(full(S)) - 1) <= TOL
        for E in (A, B, C):
            assert -TOL <= P(E) <= 1 + TOL
        assert abs(P(~A) - (1 - P(A))) <= TOL
        assert abs(P(A | B) - (P(A) + P(B) - P(A & B))) <= TOL                  # inclusion-exclusion
        assert P(A | B) <= P(A) + P(B) + TOL                                    # union bound
        assert abs(P(A | B | C) - (P(A) + P(B) + P(C) - P(A & B) - P(A & C) - P(B & C) + P(A & B & C))) <= 1e-11
        if A.is_subset(B):
            assert P(A) <= P(B) + TOL                                           # monotonicity
        assert abs(P(A - B) - (P(A) - P(A & B))) <= TOL
        assert abs(P(A) - (P(A & B) + P(A & ~B))) <= TOL                        # total probability
        if A.is_disjoint(B):
            assert abs(P(A | B) - (P(A) + P(B))) <= TOL

    @given(model_with_events())
    def test_conditional_probability_is_a_probability_measure(self, m):
        S, P, (A, B, C) = m
        assume(P(B) > 1e-9)
        cp = ConditionalProbability(P)
        assert abs(cp.compute(full(S), B) - 1) <= TOL and cp.compute(empty(S), B) == 0
        assert abs(cp.compute(A, B) * P(B) - P(A & B)) <= TOL
        assert abs(cp.compute(A, B) + cp.compute(~A, B) - 1) <= TOL
        assert abs(cp.compute(A | C, B) - (cp.compute(A, B) + cp.compute(C, B) - cp.compute(A & C, B))) <= 1e-11
        assert 0 <= cp.compute(A, B) <= 1 + TOL
        if abs(P(A & B) - P(A) * P(B)) <= 1e-15:                                # independence
            assert abs(cp.compute(A, B) - P(A)) <= 1e-11

    @given(model_with_events())
    def test_chain_rule(self, m):
        _, P, (A, B, C) = m
        assume(P(A & B) > 1e-9)
        cp = ConditionalProbability(P)
        assert abs(P(A & B & C) - P(A) * cp.compute(B, A) * cp.compute(C, A & B)) <= 1e-11


# ================================================================ Bayes
@st.composite
def bayes_models(draw):
    k = draw(st.integers(2, 6))
    raw = draw(st.lists(st.integers(1, 50), min_size=k, max_size=k))
    pri = {f"h{i}": w / sum(raw) for i, w in enumerate(raw)}
    lik = {f"h{i}": draw(st.integers(0, 100)) / 100 for i in range(k)}
    assume(sum(lik[h] * pri[h] for h in pri) > 1e-6)
    return pri, lik


@pytest.mark.property
class TestBayesProperties:
    @given(bayes_models())
    def test_posteriors_form_a_distribution(self, m):
        pri, lik = m
        post = {h: BayesTheorem.compute_from_partition(pri, lik, h) for h in pri}
        assert abs(sum(post.values()) - 1) <= 1e-12 and all(-TOL <= v <= 1 + TOL for v in post.values())
        assert all((post[h] == 0) == (lik[h] * pri[h] == 0) for h in pri)

    @given(bayes_models(), st.sampled_from([1e-3, 0.5, 2.0, 1e3]))
    def test_invariant_to_rescaling_all_likelihoods(self, m, c):
        pri, lik = m
        for h in pri:
            a = BayesTheorem.compute_from_partition(pri, lik, h)
            b = BayesTheorem.compute_from_partition(pri, {k: c * v for k, v in lik.items()}, h)
            assert abs(a - b) <= 1e-12

    @given(bayes_models())
    def test_odds_form_and_symmetry_in_labels(self, m):
        pri, lik = m
        hs = sorted(pri)
        a, b = hs[0], hs[1]
        assume(lik[b] * pri[b] > 0 and lik[a] * pri[a] > 0)
        pa, pb = (BayesTheorem.compute_from_partition(pri, lik, h) for h in (a, b))
        assert abs(pa / pb - (lik[a] / lik[b]) * (pri[a] / pri[b])) <= 1e-9 * (1 + pa / pb)   # posterior odds = LR * prior odds
        rename = {h: f"z{i}" for i, h in enumerate(reversed(hs))}
        pri2, lik2 = {rename[h]: v for h, v in pri.items()}, {rename[h]: v for h, v in lik.items()}
        assert BayesTheorem.compute_from_partition(pri2, lik2, rename[a]) == pytest.approx(pa, abs=1e-15)

    @given(spaces_and_measure(), st.data())
    def test_events_api_matches_partition_api_and_definition(self, sm, data):
        S, P = sm
        H = Event("H", data.draw(subsets(S)), S)
        E = Event("E", data.draw(subsets(S)), S)
        assume(P(E) > 1e-9 and P(H) > 1e-9)
        bt = BayesTheorem(P)
        direct = P(H & E) / P(E)
        assert abs(bt.compute_from_events(H, E) - direct) <= 1e-12
        pri = {"H": P(H), "notH": P(~H)}
        lik = {"H": P(H & E) / P(H), "notH": (P(~H & E) / P(~H)) if P(~H) > 1e-9 else 0.0}
        assume(abs(sum(pri.values()) - 1) < 1e-9)
        assert abs(BayesTheorem.compute_from_partition(pri, lik, "H") - direct) <= 1e-9


# ================================================================ discrete distributions
@pytest.mark.property
class TestDiscreteDistributionLaws:
    @given(st.integers(1, 60), st.floats(0, 1))
    def test_binomial_pmf_is_a_distribution_with_matching_moments(self, n, p):
        d = BinomialDistribution(n, p)
        pm = [d.pmf(k) for k in range(n + 1)]
        mean = sum(k * x for k, x in enumerate(pm))
        var = sum((k - mean) ** 2 * x for k, x in enumerate(pm))
        assert abs(sum(pm) - 1) <= 1e-12 and min(pm) >= 0
        assert abs(mean - d.expected_value()) <= 1e-10 and abs(var - d.variance()) <= 1e-9
        run = 0.0
        for k in range(n + 1):
            run += pm[k]
            assert abs(d.cdf(k) - run) <= 1e-12                              # cdf == cumulative pmf
        assert all(d.cdf(k) <= d.cdf(k + 1) + 1e-12 for k in range(n))      # monotone (rounding-level slack)

    @given(st.integers(1, 80), st.floats(0, 1))
    def test_binomial_cdf_stays_inside_unit_interval_and_is_monotone_exactly(self, n, p):
        """BUG: cdf(n-1) can come out as 1.0000000000000013 and then 'drops' to 1.0 at k=n."""
        d = BinomialDistribution(n, p)
        vals = [d.cdf(k) for k in range(n + 2)]
        assert all(0.0 <= v <= 1.0 for v in vals), max(vals)
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    @given(st.integers(1, 40), st.floats(1e-3, 0.999), st.data())
    def test_binomial_symmetry_recurrence_and_bernoulli_special_case(self, n, p, data):
        d, dq = BinomialDistribution(n, p), BinomialDistribution(n, 1 - p)
        k = data.draw(st.integers(0, n))
        assert d.pmf(k) == pytest.approx(dq.pmf(n - k), rel=1e-12, abs=1e-300)       # success/failure swap
        if 0 < p < 1 and k < n:
            assert d.pmf(k + 1) / d.pmf(k) == pytest.approx((n - k) / (k + 1) * p / (1 - p), rel=1e-10)
        b, bi = BernoulliDistribution(p), BinomialDistribution(1, p)
        assert all(b.pmf(j) == pytest.approx(bi.pmf(j)) for j in (0, 1)) and b.variance() == pytest.approx(bi.variance())

    @given(st.integers(1, 25), st.integers(1, 25), st.floats(0.01, 0.99), st.data())
    def test_binomial_convolution_additivity(self, n1, n2, p, data):
        k = data.draw(st.integers(0, n1 + n2))
        a, b, s = BinomialDistribution(n1, p), BinomialDistribution(n2, p), BinomialDistribution(n1 + n2, p)
        conv = sum(a.pmf(i) * b.pmf(k - i) for i in range(max(0, k - n2), min(n1, k) + 1))
        assert conv == pytest.approx(s.pmf(k), rel=1e-10)

    @given(st.floats(0.05, 40), st.floats(0.05, 40), st.integers(0, 40))
    def test_poisson_recurrence_and_additivity(self, a, b, k):
        pa, pb, ps = PoissonDistribution(a), PoissonDistribution(b), PoissonDistribution(a + b)
        assert pa.pmf(k + 1) == pytest.approx(pa.pmf(k) * a / (k + 1), rel=1e-10)
        conv = sum(pa.pmf(i) * pb.pmf(k - i) for i in range(k + 1))
        assert conv == pytest.approx(ps.pmf(k), rel=1e-9)
        assert pa.expected_value() == pa.variance()
        assert pa.cdf(k) <= pa.cdf(k + 1) + 1e-15

    @given(st.floats(0.05, 0.95), st.integers(1, 30), st.integers(1, 30))
    def test_geometric_memorylessness_and_normalisation(self, p, m, n):
        assume((1 - p) ** (m + n) > 1e-6)           # 1-cdf cancels below ~1e-10; keep the oracle meaningful
        d = GeometricDistribution(p)
        surv = lambda k: 1 - d.cdf(k)
        assert surv(m + n) / surv(m) == pytest.approx(surv(n), rel=1e-9)
        assert d.pmf(m) == pytest.approx(d.cdf(m) - (d.cdf(m - 1) if m > 1 else 0), rel=1e-9, abs=1e-12)
        assert d.expected_value() * p == pytest.approx(1.0) and d.variance() == pytest.approx((1 - p) / p ** 2)


# ================================================================ continuous distributions
def simpson(f, a, b, n=4000):
    h = (b - a) / n
    s = f(a) + f(b) + sum((4 if i % 2 else 2) * f(a + i * h) for i in range(1, n))
    return s * h / 3


@pytest.mark.property
class TestContinuousDistributionLaws:
    @given(st.floats(-50, 50), st.floats(0.05, 20), st.floats(-8, 8))
    def test_gaussian_location_scale_invariance_and_symmetry(self, mu, sigma, z):
        d, std = GaussianDistribution(mu, sigma), GaussianDistribution(0, 1)
        x = mu + z * sigma
        assert d.cdf(x) == pytest.approx(std.cdf(z), abs=1e-12)
        assert d.pdf(x) == pytest.approx(std.pdf(z) / sigma, rel=1e-9)
        assert d.pdf(mu + z * sigma) == pytest.approx(d.pdf(mu - z * sigma), rel=1e-9)
        assert d.cdf(mu + z * sigma) + d.cdf(mu - z * sigma) == pytest.approx(1.0, abs=1e-12)
        assert d.variance() == pytest.approx(sigma ** 2) and d.expected_value() == mu

    @given(st.floats(0.02, 30), st.floats(0, 20), st.floats(0, 20))
    def test_exponential_scale_and_memorylessness(self, lam, s, t):
        d, std = ExponentialDistribution(lam), ExponentialDistribution(1.0)
        assert d.cdf(s) == pytest.approx(std.cdf(lam * s), rel=1e-9, abs=1e-300)
        assert d.pdf(s) == pytest.approx(lam * std.pdf(lam * s), rel=1e-9, abs=1e-300)
        sv = lambda x: 1 - d.cdf(x)
        assume(lam * (s + t) < 8)                   # keep 1-cdf well above rounding noise
        assume(sv(s) > 1e-9)
        assert sv(s + t) / sv(s) == pytest.approx(sv(t), rel=1e-6, abs=1e-12)
        assert d.expected_value() * lam == pytest.approx(1.0) and d.variance() * lam ** 2 == pytest.approx(1.0)

    @given(st.floats(-20, 20), st.floats(0.1, 20), st.floats(0, 1))
    def test_uniform_affine_invariance_and_moments(self, a, w, u):
        d, std = ContinuousUniformDistribution(a, a + w), ContinuousUniformDistribution(0, 1)
        assert d.cdf(a + u * w) == pytest.approx(std.cdf(u), abs=1e-9)
        assert d.pdf(a + u * w) == pytest.approx(std.pdf(u) / w, rel=1e-9)
        assert d.expected_value() == pytest.approx(a + w / 2) and d.variance() == pytest.approx(w * w / 12)

    @given(st.floats(0.1, 10), st.floats(0, 8))
    def test_gamma_with_shape_one_is_exponential(self, beta, x):
        g, e = GammaDistribution(1, beta), ExponentialDistribution(beta)
        assert g.pdf(x) == pytest.approx(e.pdf(x), rel=1e-12, abs=1e-300)
        assert g.cdf(x) == pytest.approx(e.cdf(x), abs=1e-6)       # Simpson(n=1000) error ~ (beta*x/n)^4

    @given(st.floats(1, 12), st.floats(0.2, 5), st.floats(0.01, 10))
    def test_gamma_rate_scaling(self, alpha, beta, x):
        assert GammaDistribution(alpha, beta).pdf(x) == pytest.approx(beta * GammaDistribution(alpha, 1).pdf(beta * x), rel=1e-9, abs=1e-300)
        assert GammaDistribution(alpha, beta).expected_value() == pytest.approx(alpha / beta)

    @given(st.integers(1, 15).map(float), st.integers(1, 15).map(float), st.floats(0.01, 0.99))
    def test_beta_reflection_symmetry_and_uniform_special_case(self, a, b, x):
        assert BetaDistribution(a, b).pdf(x) == pytest.approx(BetaDistribution(b, a).pdf(1 - x), rel=1e-9, abs=1e-300)
        assert BetaDistribution(a, b).cdf(x) + BetaDistribution(b, a).cdf(1 - x) == pytest.approx(1.0, abs=1e-7)
        assert BetaDistribution(1, 1).pdf(x) == pytest.approx(1.0) and BetaDistribution(1, 1).cdf(x) == pytest.approx(x, abs=1e-12)

    @given(st.integers(1, 8).map(float), st.floats(0.3, 3))
    def test_gamma_moments_match_integration_of_pdf(self, alpha, beta):
        d, hi = GammaDistribution(alpha, beta), (alpha + 40) / beta
        m0 = simpson(d.pdf, 0, hi)
        m1 = simpson(lambda x: x * d.pdf(x), 0, hi)
        m2 = simpson(lambda x: x * x * d.pdf(x), 0, hi)
        assert m0 == pytest.approx(1.0, abs=1e-6)
        assert m1 == pytest.approx(d.expected_value(), rel=1e-6)
        assert m2 - m1 ** 2 == pytest.approx(d.variance(), rel=1e-5)

    @given(st.integers(1, 12).map(float), st.integers(1, 12).map(float))
    def test_beta_moments_match_integration_of_pdf(self, a, b):
        d = BetaDistribution(a, b)
        m1 = simpson(lambda x: x * d.pdf(x), 0, 1)
        m2 = simpson(lambda x: x * x * d.pdf(x), 0, 1)
        assert simpson(d.pdf, 0, 1) == pytest.approx(1.0, abs=1e-6)
        assert m1 == pytest.approx(d.expected_value(), rel=1e-6) and m2 - m1 ** 2 == pytest.approx(d.variance(), rel=1e-5)

    @given(st.floats(-5, 5), st.floats(-5, 5))
    def test_cdf_is_pdf_integral_and_monotone(self, x, y):
        lo, hi = min(x, y), max(x, y)
        d = GaussianDistribution(0.5, 1.5)
        assert d.cdf(hi) - d.cdf(lo) == pytest.approx(simpson(d.pdf, lo, hi, 2000), abs=1e-9)
        assert d.cdf(lo) <= d.cdf(hi)
        u = ContinuousUniformDistribution(-3, 4)            # pdf has jumps: compare with the closed form instead
        assert u.cdf(hi) - u.cdf(lo) == pytest.approx((min(hi, 4) - max(lo, -3)) / 7 if min(hi, 4) > max(lo, -3) else 0.0, abs=1e-12)
