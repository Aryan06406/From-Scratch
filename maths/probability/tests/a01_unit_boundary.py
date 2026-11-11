"""Unit + boundary + edge-case tests for maths.probability (foundations, Bayes, distributions)."""
import math

import numpy as np
import pytest

from maths.probability.core.m00_probability_foundations import (
    ConditionalProbability, Event, ProbabilityMeasure, RandomVariable, SampleSpace)
from maths.probability.core.m01_bayes_theorem import BayesTheorem
from maths.probability.core.m02_distributions import (
    BernoulliDistribution, BetaDistribution, BinomialDistribution, ContinuousUniformDistribution,
    ExponentialDistribution, GammaDistribution, GaussianDistribution, GeometricDistribution,
    PoissonDistribution)

NAN, INF = float("nan"), float("inf")

# foundations
@pytest.mark.boundary
class TestSampleSpaceAndEvents:
    def test_empty_sample_space_rejected(self):
        for empty in ([], (), set(), iter(())):
            with pytest.raises(ValueError):
                SampleSpace(empty)

    def test_duplicates_collapse_and_equality_ignores_order(self):
        assert len(SampleSpace([1, 1, 2, 2, 2])) == 2
        assert SampleSpace([1, 2, 3]) == SampleSpace([3, 2, 1]) != SampleSpace([1, 2])

    def test_unhashable_outcomes_rejected(self):
        with pytest.raises(TypeError):
            SampleSpace([[1], [2]])

    def test_event_outside_space_rejected(self):
        with pytest.raises(ValueError):
            Event("bad", [1, 99], SampleSpace([1, 2]))

    def test_empty_and_full_events_and_complements(self):
        S = SampleSpace(range(4))
        empty, full = Event("0", [], S), Event("1", range(4), S)
        assert empty.is_empty() and ~empty == full and ~full == empty and ~~full == full
        assert full - full == empty and empty | full == full and empty & full == empty

    def test_cross_space_operations_rejected(self):
        A, B = Event("a", [1], SampleSpace([1, 2])), Event("b", [1], SampleSpace([1, 2, 3]))
        for op in (lambda: A | B, lambda: A & B, lambda: A - B):
            with pytest.raises(ValueError):
                op()

    def test_same_outcomes_different_space_not_equal(self):
        assert Event("a", [1], SampleSpace([1, 2])) != Event("a", [1], SampleSpace([1, 2, 3]))

    def test_copy_is_independent_and_renamable(self):
        S = SampleSpace([1, 2])
        e = Event("e", [1], S)
        c = e.copy("c")
        c.outcomes.add(2)
        assert e.outcomes == {1} and c.name == "c"

    def test_from_predicted(self):
        S = SampleSpace(range(10))
        assert Event.from_predicted("even", S, lambda x: x % 2 == 0).cardinality() == 5

    def test_generator_outcomes_consumed_once(self):
        S = SampleSpace(range(3))
        assert Event("g", (x for x in range(2)), S).cardinality() == 2

@pytest.mark.boundary
class TestProbabilityMeasure:
    S = SampleSpace(["a", "b", "c"])

    @pytest.mark.parametrize("w", [
        {"a": 0.5, "b": 0.5},                          # missing outcome
        {"a": 0.5, "b": 0.5, "c": 0.0, "d": 0.0},      # extra outcome
        {"a": -0.1, "b": 0.6, "c": 0.5},               # negative
        {"a": 0.5, "b": 0.5, "c": 0.5},                # sum != 1
        {"a": 1 + 1e-8, "b": 0.0, "c": 0.0},           # outside 1e-9 tolerance
    ], ids=["missing", "extra", "negative", "sum>1", "tol"])
    def test_invalid_weights(self, w):
        with pytest.raises(ValueError):
            ProbabilityMeasure(self.S, w)

    @pytest.mark.parametrize("bad", [NAN, INF, -INF], ids=["nan", "inf", "-inf"])
    def test_non_finite_weights_rejected(self, bad):
        with pytest.raises(ValueError):
            ProbabilityMeasure(self.S, {"a": bad, "b": 0.5, "c": 0.5})

    def test_weights_dict_is_not_aliased(self):
        w = {"a": 0.2, "b": 0.3, "c": 0.5}
        P = ProbabilityMeasure(self.S, w)
        w["a"] = 5.0
        assert P(Event("all", "abc", self.S)) == pytest.approx(1.0)

    def test_zero_probability_outcomes_allowed(self):
        P = ProbabilityMeasure(self.S, {"a": 1.0, "b": 0.0, "c": 0.0})
        assert P(Event("b", ["b"], self.S)) == 0

    @pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 1000])
    def test_uniform_total_mass(self, n):
        S = SampleSpace(range(n))
        P = ProbabilityMeasure(S)
        assert P(Event("all", range(n), S)) == pytest.approx(1.0, abs=n * 2.3e-16)
        assert P(Event("none", [], S)) == 0

    def test_event_from_other_space_rejected(self):
        P = ProbabilityMeasure(self.S)
        with pytest.raises(ValueError):
            P(Event("x", [1], SampleSpace([1, 2])))

    def test_tolerance_policies_are_consistent_between_modules(self):
        priors = {"a": 0.5 + 5e-8, "b": 0.5}
        accepted_by_measure = True
        try:
            ProbabilityMeasure(SampleSpace(["a", "b"]), dict(priors))
        except ValueError:
            accepted_by_measure = False
        accepted_by_bayes = True
        try:
            BayesTheorem.compute_from_partition(priors, {"a": 0.3, "b": 0.7}, "a")
        except ValueError:
            accepted_by_bayes = False
        assert accepted_by_measure == accepted_by_bayes

@pytest.mark.boundary
class TestConditionalAndRandomVariable:
    def setup_method(self):
        self.S = SampleSpace(range(6))
        self.P = ProbabilityMeasure(self.S)
        self.C = ConditionalProbability(self.P)

    def test_basic_conditionals(self):
        even, big = Event.from_predicted("e", self.S, lambda x: x % 2 == 0), Event.from_predicted("b", self.S, lambda x: x >= 3)
        assert self.C.compute(even, big) == pytest.approx(1 / 3)
        assert self.C.compute(even, even) == 1 and self.C.compute(Event("n", [], self.S), even) == 0
        assert self.C.compute(even, Event("all", range(6), self.S)) == pytest.approx(0.5)

    def test_zero_probability_condition_rejected(self):
        with pytest.raises(ValueError):
            self.C.compute(Event("a", [1], self.S), Event("n", [], self.S))

    def test_random_variable_domain_and_events(self):
        X = RandomVariable(self.S, lambda w: w % 3)
        with pytest.raises(ValueError):
            X(99)
        assert X.event_equals(1).outcomes == {1, 4}
        assert X.event_less_than_or_equal(0).outcomes == {0, 3}
        assert X.event_equals(7).is_empty()


@pytest.mark.boundary
class TestBayes:
    def test_textbook_medical_test(self):
        post = BayesTheorem.compute_from_partition({"D": 0.01, "H": 0.99}, {"D": 0.95, "H": 0.05}, "D")
        assert post == pytest.approx(0.0095 / (0.0095 + 0.0495))

    def test_events_api_matches_closed_form(self):
        S = SampleSpace(["dh", "dn", "hh", "hn"])
        P = ProbabilityMeasure(S, {"dh": 0.0095, "dn": 0.0005, "hh": 0.0495, "hn": 0.9405})
        D, T = Event("D", ["dh", "dn"], S), Event("T", ["dh", "hh"], S)
        assert BayesTheorem(P).compute_from_events(D, T) == pytest.approx(0.0095 / 0.059)

    def test_zero_prior_hypothesis_has_zero_posterior(self):
        S = SampleSpace([1, 2, 3])
        P = ProbabilityMeasure(S, {1: 0.5, 2: 0.5, 3: 0.0})
        assert BayesTheorem(P).compute_from_events(Event("H", [3], S), Event("E", [1, 3], S)) == 0

    def test_zero_evidence_rejected(self):
        S = SampleSpace([1, 2])
        P = ProbabilityMeasure(S)
        with pytest.raises(ValueError):
            BayesTheorem(P).compute_from_events(Event("H", [1], S), Event("E", [], S))
        with pytest.raises(ValueError):
            BayesTheorem.compute_from_partition({"a": 0.5, "b": 0.5}, {"a": 0.0, "b": 0.0}, "a")

    @pytest.mark.parametrize("priors,likes,target", [
        ({"a": 0.5, "b": 0.5}, {"a": 0.5}, "a"),                          # key mismatch
        ({"a": 0.5, "b": 0.4}, {"a": 0.5, "b": 0.5}, "a"),                # priors != 1
        ({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}, "zzz"),              # unknown target
        ({"a": 1.5, "b": -0.5}, {"a": 0.5, "b": 0.5}, "a"),               # negative prior
        ({"a": 0.5, "b": 0.5}, {"a": 5.0, "b": 0.5}, "a"),                # likelihood > 1
        ({"a": 0.5, "b": 0.5}, {"a": -0.1, "b": 0.5}, "a"),               # likelihood < 0
        ({"a": 0.5, "b": 0.5}, {"a": NAN, "b": 0.5}, "a"),                # NaN likelihood
        ({"a": NAN, "b": 0.5}, {"a": 0.5, "b": 0.5}, "a"),                # NaN prior
    ], ids=["keys", "sum", "target", "neg-prior", "like>1", "like<0", "nan-like", "nan-prior"])
    def test_partition_input_validation(self, priors, likes, target):
        with pytest.raises(ValueError):
            BayesTheorem.compute_from_partition(priors, likes, target)

# distributions
INVALID_PARAMS = [
    (BernoulliDistribution, (-0.1,)), (BernoulliDistribution, (1.1,)), (BernoulliDistribution, (NAN,)),
    (BinomialDistribution, (0, 0.5)), 
    (BinomialDistribution, (-3, 0.5)), (BinomialDistribution, (5, 1.5)), (BinomialDistribution, (5, NAN)), 
    (BinomialDistribution, (2.5, 0.5)),
    (GeometricDistribution, (0.0,)), (GeometricDistribution, (-0.1,)), (GeometricDistribution, (1.5,)),
    (GeometricDistribution, (NAN,)),
    (PoissonDistribution, (0,)), (PoissonDistribution, (-1,)), (PoissonDistribution, (NAN,)),
    (ContinuousUniformDistribution, (1, 1)), (ContinuousUniformDistribution, (2, 1)),
    (ContinuousUniformDistribution, (NAN, 1)), (ContinuousUniformDistribution, (0, NAN)),
    (ExponentialDistribution, (0,)), (ExponentialDistribution, (-1,)), (ExponentialDistribution, (NAN,)),
    (GaussianDistribution, (0, 0)), (GaussianDistribution, (0, -1)), (GaussianDistribution, (0, NAN)),
    (GaussianDistribution, (NAN, 1)),
    (GammaDistribution, (0, 1)), (GammaDistribution, (1, 0)), (GammaDistribution, (-1, 1)),
    (GammaDistribution, (NAN, 1)), (GammaDistribution, (1, NAN)),
    (BetaDistribution, (0, 1)), (BetaDistribution, (1, 0)), (BetaDistribution, (NAN, 1)), (BetaDistribution, (1, NAN))
]

@pytest.mark.boundary
class TestDistributionParameterValidation:
    @pytest.mark.parametrize("cls,args", INVALID_PARAMS,
                             ids=[f"{c.__name__[:-12]}{a}" for c, a in INVALID_PARAMS])
    def test_invalid_parameters_are_rejected(self, cls, args):
        with pytest.raises(ValueError):
            cls(*args)

@pytest.mark.boundary
class TestDiscreteBoundaries:
    def test_bernoulli_degenerate(self):
        for p in (0.0, 1.0):
            d = BernoulliDistribution(p)
            assert d.pmf(1) == p and d.pmf(0) == 1 - p and d.variance() == 0 and d.std_dev() == 0
        d = BernoulliDistribution(0.3)
        assert [d.cdf(x) for x in (-1e-9, 0, 0.5, 1, 7)] == [0.0, 0.7, 0.7, 1.0, 1.0]
        for bad in (2, -1, 0.5):
            with pytest.raises(ValueError):
                d.pmf(bad)

    @pytest.mark.parametrize("p,k,expect", [(0.0, 0, 1.0), (0.0, 3, 0.0), (1.0, 5, 1.0), (1.0, 4, 0.0)])
    def test_binomial_degenerate_probabilities(self, p, k, expect):
        assert BinomialDistribution(5, p).pmf(k) == expect

    def test_binomial_support_edges(self):
        d = BinomialDistribution(10, 0.5)
        assert d.pmf(0) == d.pmf(10) == 0.5 ** 10
        assert d.cdf(-1) == 0 and d.cdf(10) == 1 and d.cdf(99) == 1
        for bad in (-1, 11, 2.5, "3"):
            with pytest.raises((ValueError, TypeError)):
                d.pmf(bad)

    @pytest.mark.parametrize("cdf_arg", [2.5, 2.999, 0.2, 9.5])
    def test_binomial_cdf_accepts_real_arguments(self, cdf_arg):
        d = BinomialDistribution(10, 0.4)
        assert d.cdf(cdf_arg) == pytest.approx(d.cdf(math.floor(cdf_arg)))

    @pytest.mark.parametrize("make", [lambda: GeometricDistribution(0.3), lambda: PoissonDistribution(2.0),
                                      lambda: BinomialDistribution(5, 0.3), lambda: BernoulliDistribution(0.3)],
                             ids=["geometric", "poisson", "binomial", "bernoulli"])
    def test_cdf_is_zero_below_support(self, make):
        d = make()
        assert d.cdf(-1) == 0.0 and d.cdf(-100) == 0.0
        if isinstance(d, GeometricDistribution):
            assert d.cdf(0) == 0.0

    @pytest.mark.parametrize("make", [lambda: GeometricDistribution(0.3), lambda: PoissonDistribution(2.0)],
                             ids=["geometric", "poisson"])
    def test_cdf_accepts_non_integer_reals(self, make):
        d = make()
        assert d.cdf(3.7) == pytest.approx(d.cdf(3))

    @pytest.mark.parametrize("make", [lambda: BinomialDistribution(10, 0.4), lambda: GeometricDistribution(0.3),
                                      lambda: PoissonDistribution(2.0)], ids=["binomial", "geometric", "poisson"])
    def test_numpy_integers_accepted(self, make):
        d = make()
        assert d.pmf(np.int64(3)) == pytest.approx(d.pmf(3))

    def test_geometric_edges(self):
        d = GeometricDistribution(1.0)
        assert d.pmf(1) == 1.0 and d.pmf(2) == 0.0 and d.variance() == 0 and d.expected_value() == 1
        g = GeometricDistribution(0.5)
        assert g.pmf(1) == 0.5 and g.cdf(1) == 0.5
        with pytest.raises(ValueError):
            g.pmf(0)

    def test_poisson_edges(self):
        d = PoissonDistribution(1e-300)
        assert d.pmf(0) == 1.0 and d.pmf(1) == pytest.approx(1e-300)
        e = PoissonDistribution(3.0)
        assert e.pmf(0) == pytest.approx(math.exp(-3)) and e.variance() == e.expected_value() == 3.0
        with pytest.raises(ValueError):
            e.pmf(-1)

@pytest.mark.boundary
class TestContinuousBoundaries:
    def test_uniform_edges(self):
        d = ContinuousUniformDistribution(-1, 3)
        assert d.pdf(-1) == d.pdf(3) == 0.25 and d.pdf(-1.0001) == 0 == d.pdf(3.0001)
        assert d.cdf(-1) == 0 and d.cdf(3) == 1 and d.cdf(1) == 0.5
        assert d.expected_value() == 1 and d.variance() == pytest.approx(16 / 12)

    def test_exponential_edges(self):
        d = ExponentialDistribution(2.0)
        assert d.pdf(0) == 2.0 and d.pdf(-1e-12) == 0 and d.cdf(0) == 0 and d.cdf(-5) == 0
        assert d.cdf(1e6) == 1.0 and d.pdf(1e6) == 0.0

    def test_gaussian_edges(self):
        d = GaussianDistribution(0, 1)
        assert d.cdf(0) == 0.5 and d.cdf(INF) == 1.0 and d.cdf(-INF) == 0.0 and d.pdf(INF) == 0.0
        assert d.pdf(0) == pytest.approx(1 / math.sqrt(2 * math.pi))
        assert math.isnan(d.pdf(NAN))

    @pytest.mark.parametrize("x", [1e155, -1e155, 1e200, 1e300])
    def test_gaussian_pdf_far_tail_is_zero_not_an_exception(self, x):
        assert GaussianDistribution(0, 1).pdf(x) == 0.0

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 0.99])
    def test_gamma_with_shape_below_one(self, alpha):
        d = GammaDistribution(alpha, 1.0)
        assert d.pdf(0) == INF or d.pdf(0) > 1e300
        assert 0.0 < d.cdf(1.0) < 1.0

    @pytest.mark.parametrize("a,b", [(0.5, 0.5), (0.5, 2.0), (2.0, 0.5)], ids=["arcsine", "a<1", "b<1"])
    def test_beta_with_shape_below_one(self, a, b):
        d = BetaDistribution(a, b)
        assert 0.0 < d.cdf(0.5) < 1.0
        assert d.cdf(1.0) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("make,x", [
        (lambda: GammaDistribution(2, 1), 0), (lambda: GammaDistribution(1, 3), 0),
        (lambda: BetaDistribution(2, 3), 0), (lambda: BetaDistribution(2, 3), 1), (lambda: BetaDistribution(1, 1), 0.5)])
    def test_endpoint_values_regular_shapes(self, make, x):
        d = make()
        assert d.cdf(0) == 0 and d.pdf(-1) == 0
        if isinstance(d, BetaDistribution):
            assert d.cdf(1) == pytest.approx(1.0) and d.pdf(2) == 0

    def test_gamma_beta_cdf_boundaries(self):
        assert GammaDistribution(2, 1).cdf(-1) == 0
        assert BetaDistribution(2, 2).cdf(-1) == 0 and BetaDistribution(2, 2).cdf(2) == 1

    @pytest.mark.parametrize("make", [
        lambda: ExponentialDistribution(0.5), lambda: GaussianDistribution(1, 2), lambda: GammaDistribution(3, 2),
        lambda: BetaDistribution(2, 5), lambda: ContinuousUniformDistribution(0, 4), lambda: PoissonDistribution(4.0),
        lambda: BinomialDistribution(9, 0.3), lambda: GeometricDistribution(0.2), lambda: BernoulliDistribution(0.4)])
    def test_std_dev_is_sqrt_variance(self, make):
        d = make()
        assert d.std_dev() == pytest.approx(math.sqrt(d.variance()), rel=1e-15)
