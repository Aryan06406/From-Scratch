import math
import pytest

from maths.linear_algebra.core.m01_vector_ops import Vector
from maths.linear_algebra.core.m03_linear_transformation import LinearTransformation
from maths.linear_algebra.core.m04_projection import Projection


def test_projection():
    u = Vector([3, 4])
    v = Vector([1, 0])

    assert math.isclose(Projection.scalar_projection(u, v), 3)
    assert Projection.vector_projection(u, v) == Vector([3, 0])
    assert Projection.rejection(u, v) == Vector([0, 4])
    assert Projection.orthogonal_component(u, v) == Vector([0, 4])

    matrix = Projection.projection_matrix(v)
    assert matrix == LinearTransformation([[1.0, 0.0], [0.0, 0.0]])


def test_projection_validation():
    with pytest.raises(ValueError):
        Projection.vector_projection(Vector([1, 2]), Vector([0, 0]))



