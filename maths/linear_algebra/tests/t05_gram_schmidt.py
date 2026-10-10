import math
import pytest

from maths.linear_algebra.core.m01_vector_ops import Vector
from maths.linear_algebra.core.m05_gram_schmidt import GramSchmidt

def test_gram_schmidt():
    vectors = [Vector([1, 1]), Vector([1, 0])]
    orthogonal = GramSchmidt.orthogonal_basis(vectors)
    orthonormal = GramSchmidt.orthonormal_basis(vectors)

    assert len(orthogonal) == 2
    assert GramSchmidt.is_orthogonal_basis(orthogonal)
    assert len(orthonormal) == 2
    assert GramSchmidt.is_orthonormal_basis(orthonormal)
    assert all(math.isclose(v.magnitude(), 1.0) for v in orthonormal)

def test_gram_schmidt_drops_dependent_vectors():
    basis = GramSchmidt.orthogonal_basis(
        [Vector([1, 2]), Vector([2, 4]), Vector([0, 1])]
    )
    assert len(basis) == 2

def test_gram_schmidt_empty_input():
    with pytest.raises(ValueError):
        GramSchmidt.orthogonal_basis([])