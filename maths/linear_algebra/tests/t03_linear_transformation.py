import math
import pytest

from maths.linear_algebra.core.m01_vector_ops import Vector
from maths.linear_algebra.core.m03_linear_transformation import LinearTransformation


def test_matrix_arithmetic_and_shape():
    a = LinearTransformation([[1, 2], [3, 4]])
    b = LinearTransformation([[5, 6], [7, 8]])

    assert a.shape == (2, 2)
    assert a + b == LinearTransformation([[6, 8], [10, 12]])
    assert b - a == LinearTransformation([[4, 4], [4, 4]])
    assert a * 2 == LinearTransformation([[2, 4], [6, 8]])
    assert 2 * a == LinearTransformation([[2, 4], [6, 8]])
    assert a * b == LinearTransformation([[19, 22], [43, 50]])


def test_matrix_vector_and_transpose():
    a = LinearTransformation([[1, 2], [3, 4]])
    assert a.apply_transformation(Vector([5, 6])) == Vector([17, 39])
    assert a.transpose() == LinearTransformation([[1, 3], [2, 4]])


def test_trace_determinant_and_inverse():
    a = LinearTransformation([[4, 7], [2, 6]])

    assert a.trace() == 10
    assert math.isclose(a.determinant(), 10)
    assert math.isclose(a.determinant_optimised(), 10)

    expected = LinearTransformation([[0.6, -0.7], [-0.2, 0.4]])
    inv = a.inverse_optimised()
    for i in range(2):
        for j in range(2):
            assert math.isclose(inv[i][j], expected[i][j], abs_tol=1e-9)

    inv2 = a.inverse()
    for i in range(2):
        for j in range(2):
            assert math.isclose(inv2[i][j], expected[i][j], abs_tol=1e-9)


def test_matrix_properties_and_factories():
    identity = LinearTransformation.identity(3)
    zeros = LinearTransformation.zeros(2, 3)

    assert identity.is_invertible()
    assert identity.is_symmetric()
    assert identity.is_orthogonal()
    assert zeros.shape == (2, 3)
    assert not zeros.is_invertible()


def test_matrix_validation():
    with pytest.raises(ValueError):
        LinearTransformation([])

    with pytest.raises(ValueError):
        LinearTransformation([[1, 2], [3]])

    a = LinearTransformation([[1, 2]])
    with pytest.raises(ValueError):
        a.trace()

    with pytest.raises(ValueError):
        a * LinearTransformation([[1, 2]])
