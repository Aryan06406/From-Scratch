import math

from maths.linear_algebra.core.m01_vector_ops import Vector
from maths.linear_algebra.core.m03_linear_transformation import LinearTransformation
from maths.linear_algebra.core.m06_fundamental_subspaces import FundamentalSubspaces


def test_rank_nullity_and_spaces():
    a = LinearTransformation([
        [1, 2, 3],
        [2, 4, 6],
    ])
    fs = FundamentalSubspaces(a)

    assert fs.rank() == 1
    assert fs.nullity() == 2

    column_space = fs.column_space()
    assert column_space == [Vector([1, 2])]

    row_space = fs.row_space()
    assert len(row_space) == 1
    assert row_space[0] == Vector([1, 2, 3])

    null_space = fs.null_space()
    assert len(null_space) == 2
    for v in null_space:
        assert a.apply_transformation(v).magnitude() < 1e-9


def test_left_null_space():
    a = LinearTransformation([
        [1, 0],
        [0, 0],
    ])
    fs = FundamentalSubspaces(a)

    left = fs.left_null_space()
    assert len(left) == 1
    assert math.isclose(left[0][0], 0)
    assert math.isclose(left[0][1], 1)
