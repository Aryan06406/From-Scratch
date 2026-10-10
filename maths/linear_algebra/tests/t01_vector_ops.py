import math
import pytest

from maths.linear_algebra.core.m01_vector_ops import Vector


def test_vector_arithmetic_and_scalars():
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])

    assert a + b == Vector([5, 7, 9])
    assert b - a == Vector([3, 3, 3])
    assert 2 * a == Vector([2, 4, 6])
    assert a * 3 == Vector([3, 6, 9])
    assert a / 2 == Vector([0.5, 1, 1.5])
    assert a // 2 == Vector([0, 1, 1])
    assert -a == Vector([-1, -2, -3])


def test_vector_products_and_geometry():
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])

    assert a.dot_product(b) == 32
    assert a.cross_product(b) == Vector([-3, 6, -3])
    assert a.hadamard_product(b) == Vector([4, 10, 18])
    assert math.isclose(a.magnitude(), math.sqrt(14))
    assert math.isclose(a.distance_between_vectors(b), math.sqrt(27))


def test_unit_vector_and_angle():
    a = Vector([1, 0])
    b = Vector([0, 1])

    assert a.unit_vector() == a
    assert math.isclose(a.angle_between_vectors(b), 90.0)
    assert a.is_orthogonal(b)
    assert Vector([2, 4]).is_parallel(Vector([1, 2]))


def test_vector_validation():
    a = Vector([1, 2])

    with pytest.raises(ValueError):
        a + Vector([1, 2, 3])

    with pytest.raises(ZeroDivisionError):
        a / 0

    with pytest.raises(ValueError):
        Vector([1, 2]).cross_product(Vector([3, 4]))

    with pytest.raises(ZeroDivisionError):
        Vector([0, 0]).unit_vector()

    with pytest.raises(ZeroDivisionError):
        Vector([0, 0]).angle_between_vectors(Vector([1, 0]))
