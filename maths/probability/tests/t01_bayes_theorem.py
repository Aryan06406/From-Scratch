import math
import pytest

from maths.probability.core.m01_bayes_theorem import BayesTheorem

def test_bayes_theorem():
    priors = {"Disease": 0.01, "Healthy": 0.99}
    likelihoods = {"Disease": 0.95, "Healthy": 0.05}

    posterior = BayesTheorem.compute_from_partition(
        priors, likelihoods, "Disease"
    )
    assert math.isclose(posterior, 0.16101694915254236)