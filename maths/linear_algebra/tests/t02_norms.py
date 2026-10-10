import math
import pytest

from maths.linear_algebra.core.m01_vector_ops import Vector
from maths.linear_algebra.core.m02_norms import Norms


def test_common_norms():
    norms = Norms(Vector([-3, 4]))

    assert norms.l1_norm() == 7
    assert math.isclose(norms.l2_norm(), 5)
    assert norms.infinity_norm() == 4
    assert math.isclose(norms.p_norm(3), 91 ** (1 / 3))


def test_infinity_norm_of_empty_vector():
    assert Norms(Vector([])).infinity_norm() == 0


def test_p_norm_requires_positive_p():
    with pytest.raises(ValueError):
        Norms(Vector([1, 2])).p_norm(0)
