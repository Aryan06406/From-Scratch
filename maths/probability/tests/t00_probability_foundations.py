import math
import pytest

from maths.probability.core.m00_probability_foundations import (SampleSpace, Event, ProbabilityMeasure, ConditionalProbability, RandomVariable)

def test_events_and_probability():
    omega = SampleSpace({1, 2, 3, 4, 5, 6})
    even = Event("Even", {2, 4, 6}, omega)
    greater_than_3 = Event("Greater than 3", {4, 5, 6}, omega)

    assert (even & greater_than_3).outcomes == {4, 6}
    assert (even | greater_than_3).outcomes == {2, 4, 5, 6}
    assert (~even).outcomes == {1, 3, 5}

    probability = ProbabilityMeasure(omega)
    assert math.isclose(probability(even), 0.5)
    assert math.isclose(probability(even & greater_than_3), 2 / 6)


def test_conditional_probability_and_random_variable():
    omega = SampleSpace({1, 2, 3, 4, 5, 6})
    even = Event("Even", {2, 4, 6}, omega)
    greater_than_3 = Event("Greater than 3", {4, 5, 6}, omega)
    probability = ProbabilityMeasure(omega)

    conditional = ConditionalProbability(probability)
    assert math.isclose(conditional.compute(even, greater_than_3), 2 / 3)

    x = RandomVariable(omega, lambda value: value ** 2)
    assert x(3) == 9
    assert x.event_equals(4).outcomes == {2}
    assert x.event_less_than_or_equal(9).outcomes == {1, 2, 3}