"""
Unit + boundary-value + edge case tests for maths.linear_algebra.
Every test targets a specific suspected weakness (see the docstring / parametrize ids).
"""
import math

import numpy as np
import pytest

from maths.linear_algebra.core import (FundamentalSubspaces, GramSchmidt, LinearTransformation,
                                       Norms, Projection, Vector)
from maths.test_helpers import M, V

NAN, INF = float("nan"), float("inf")


# Vector
@pytest.mark.boundary
class TestVectorBoundary:
    def test_empty_vector(self):
        e = Vector([])
        assert len(e) == 0 and e.magnitude() == 0 and e.dot_product(Vector([])) == 0
        assert e + Vector([]) == Vector([])
        with pytest.raises(ZeroDivisionError):
            e.unit_vector()
        with pytest.raises(ZeroDivisionError):
            e.angle_between_vectors(Vector([]))

    def test_single_element(self):
        a = V(-3)
        assert a.magnitude() == 3 and a.unit_vector() == V(-1.0)
        assert a.angle_between_vectors(V(5)) == pytest.approx(180.0)
        assert a.is_parallel(V(5))

    def test_constructor_copies_input(self):
        src = [1, 2, 3]
        v = Vector(src)
        src[0] = 99
        assert v[0] == 1

    def test_constructor_accepts_tuple_and_generator(self):
        assert Vector((1, 2)) == Vector(x for x in (1, 2))

    @pytest.mark.parametrize("bad", ["abc", [1, "a"], [None], [[1], [2]]],
                             ids=["string", "mixed-str", "None", "nested"])
    def test_constructor_rejects_non_numeric(self, bad):
        """BUG-candidate: Vector('abc') silently becomes ['a','b','c']; errors surface much later."""
        with pytest.raises((TypeError, ValueError)):
            Vector(bad)

    def test_indexing(self):
        v = V(1, 2, 3)
        assert v[-1] == 3
        with pytest.raises(IndexError):
            v[3]

    @pytest.mark.parametrize("zero", [0, 0.0, -0.0])
    def test_division_by_zero_all_spellings(self, zero):
        with pytest.raises(ZeroDivisionError):
            V(1, 2) / zero
        with pytest.raises(ZeroDivisionError):
            V(1, 2) // zero

    @pytest.mark.parametrize("bad", ["2", None, [1, 2], 1 + 2j, Vector([1, 2])],
                             ids=["str", "None", "list", "complex", "Vector"])
    def test_non_scalar_operands_rejected(self, bad):
        v = V(1, 2)
        for op in (lambda: v * bad, lambda: bad * v, lambda: v / bad, lambda: v // bad):
            with pytest.raises(TypeError):
                op()

    @pytest.mark.parametrize("other", [5, [1, 2], None, "ab"], ids=["int", "list", "None", "str"])
    def test_add_sub_reject_non_vectors(self, other):
        with pytest.raises(TypeError):
            V(1, 2) + other
        with pytest.raises(TypeError):
            V(1, 2) - other

    @pytest.mark.parametrize("np_scalar", [np.float64(2.0), np.int64(2), np.float32(2.0)],
                             ids=["f64", "i64", "f32"])
    def test_numpy_scalar_on_left_returns_Vector(self, np_scalar):
        """BUG-candidate: numpy scalars hijack `scalar * Vector` and return an ndarray."""
        assert isinstance(np_scalar * V(1, 2), Vector)

    def test_dimension_mismatch_everywhere(self):
        a, b = V(1, 2), V(1, 2, 3)
        for f in (a.__add__, a.__sub__, a.dot_product, a.hadamard_product,
                  a.distance_between_vectors, a.cross_product):
            with pytest.raises((ValueError, TypeError)):
                f(b)
        with pytest.raises(ValueError):
            a.is_orthogonal(b)
        with pytest.raises(ValueError):
            a.angle_between_vectors(b)

    def test_cross_product_only_3d(self):
        for n in (0, 1, 2, 4):
            with pytest.raises(ValueError):
                Vector([1] * n).cross_product(Vector([1] * n))

    def test_cross_product_self_is_zero_and_basis(self):
        assert V(1, 2, 3).cross_product(V(1, 2, 3)) == V(0, 0, 0)
        assert V(1, 0, 0).cross_product(V(0, 1, 0)) == V(0, 0, 1)

    @pytest.mark.parametrize("bad", [NAN, INF], ids=["nan", "inf"])
    def test_angle_with_non_finite_input_is_not_silently_zero(self, bad):
        """BUG-candidate: clamp max(-1,min(1,nan)) swallows NaN and returns 0 degrees."""
        try:
            r = V(bad, 1).angle_between_vectors(V(1, 1))
        except (ValueError, ArithmeticError):
            return
        assert math.isnan(r), f"angle({bad}) silently returned {r}"

    def test_zero_vector_parallel_convention_is_consistent(self):
        """BUG-candidate: zero||zero is True but zero||nonzero is False."""
        z = V(0, 0)
        assert z.is_parallel(V(1, 2)) == z.is_parallel(V(0, 0)) == V(1, 2).is_parallel(z)

    def test_orthogonality_of_zero_vector(self):
        assert V(0, 0).is_orthogonal(V(3, 4))

    def test_floor_division_semantics(self):
        assert V(7, -7) // 2 == V(3, -4)           # floor, not truncation / rounding


# Norms
@pytest.mark.boundary
class TestNormsBoundary:
    def test_type_check(self):
        with pytest.raises(TypeError):
            Norms([1, 2])

    def test_empty_vector_norms_all_zero(self):
        n = Norms(Vector([]))
        assert n.l1_norm() == n.l2_norm() == n.infinity_norm() == 0
        assert n.p_norm(3) == 0

    def test_known_values_with_negatives(self):
        n = Norms(V(3, -4))
        assert (n.l1_norm(), n.l2_norm(), n.infinity_norm()) == (7, 5, 4)
        assert n.p_norm(1) == 7 and n.p_norm(2) == pytest.approx(5)

    @pytest.mark.parametrize("p", [0, -1, -0.5, NAN], ids=["0", "-1", "-0.5", "nan"])
    def test_invalid_p_rejected(self, p):
        """BUG-candidate: p=nan sails through the `p <= 0` guard and returns nan."""
        with pytest.raises(ValueError):
            Norms(V(1, 2)).p_norm(p)

    def test_p_infinity_matches_infinity_norm(self):
        """BUG-candidate: p=inf returns 1.0 (x**inf then total**0)."""
        n = Norms(V(3, 4))
        try:
            r = n.p_norm(INF)
        except (ValueError, OverflowError):
            return
        assert r == n.infinity_norm()

    def test_zero_vector(self):
        assert Norms(V(0, 0, 0)).p_norm(0.5) == 0


# Matrix
@pytest.mark.boundary
class TestMatrixConstruction:
    @pytest.mark.parametrize("bad", [[], [[]], [[1, 2], [3]], [[1], [2, 3]]],
                             ids=["empty", "empty-row", "ragged-short", "ragged-long"])
    def test_invalid_shapes(self, bad):
        with pytest.raises(ValueError):
            LinearTransformation(bad)

    def test_flat_list_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            LinearTransformation([1, 2, 3])

    def test_tuples_accepted_and_input_not_aliased(self):
        src = [[1, 2], [3, 4]]
        A = M(src)
        src[0][0] = 99
        assert A[0][0] == 1
        assert LinearTransformation(((1, 2), (3, 4))) == M([[1, 2], [3, 4]])

    def test_results_do_not_alias_operands(self):
        A = M([[1, 2], [3, 4]])
        for B in (A * 1, A + M([[0, 0], [0, 0]]), A.transpose().transpose()):
            B[0][0] = 99
            assert A[0][0] == 1

    @pytest.mark.parametrize("n,exc", [(0, ValueError), (-1, ValueError), (2.5, TypeError)])
    def test_identity_bad_sizes(self, n, exc):
        with pytest.raises(exc):
            LinearTransformation.identity(n)

    @pytest.mark.parametrize("r,c", [(0, 2), (2, 0), (0, 0)])
    def test_zeros_degenerate(self, r, c):
        with pytest.raises(ValueError):
            LinearTransformation.zeros(r, c)

    def test_equality_across_shapes_and_types(self):
        assert M([[1, 2]]) != M([[1], [2]])
        assert M([[1]]) != 1


@pytest.mark.boundary
class TestMatrixOperations:
    def test_wrong_operand_types_raise_TypeError(self):
        """BUG-candidate: `A + 5` raises AttributeError ('int' has no .shape) instead of TypeError."""
        A = M([[1, 2], [3, 4]])
        for f in (lambda: A + 5, lambda: A - 5, lambda: A + [[1, 2], [3, 4]]):
            with pytest.raises(TypeError):
                f()
        for f in (lambda: A * "x", lambda: A * Vector([1, 2]), lambda: A * [[1, 2], [3, 4]]):
            with pytest.raises(TypeError):
                f()

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            M([[1, 2]]) + M([[1], [2]])

    def test_rectangular_product_shapes(self):
        A, B = M([[1, 2, 3], [4, 5, 6]]), M([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
        assert (A * B).shape == (2, 4)
        with pytest.raises(ValueError):
            B * A

    def test_multiplication_is_not_commutative(self):
        A, B = M([[1, 2], [3, 4]]), M([[0, 1], [1, 0]])
        assert A * B != B * A

    def test_apply_transformation_rectangular_and_errors(self):
        A = M([[1, 2, 3], [4, 5, 6]])
        assert A.apply_transformation(V(1, 0, 1)) == V(4, 10)
        with pytest.raises(ValueError):
            A.apply_transformation(V(1, 2))
        with pytest.raises(TypeError):
            A.apply_transformation([1, 2, 3])

    def test_non_square_operations_rejected(self):
        A = M([[1, 2, 3], [4, 5, 6]])
        for f in (A.trace, A.determinant, A.determinant_optimised, A.inverse, A.inverse_optimised):
            with pytest.raises(ValueError):
                f()
        assert not A.is_symmetric() and not A.is_invertible() and not A.is_orthogonal()
        assert A.transpose().shape == (3, 2)


@pytest.mark.boundary
class TestMatrixSquareEdgeCases:
    @pytest.mark.parametrize("a", [4, -4, 0.5, 1e-3])
    def test_1x1_inverse(self, a):
        """BUG: adjugate path computes det([]) == 0 for the 0x0 minor, so inverse([[a]]) == [[0.0]]."""
        assert M([[a]]).inverse()[0][0] == pytest.approx(1 / a)

    @pytest.mark.parametrize("a", [4, -4, 0.5])
    def test_1x1_inverse_optimised(self, a):
        assert M([[a]]).inverse_optimised()[0][0] == pytest.approx(1 / a)

    def test_determinant_of_empty_matrix_is_one(self):
        """Root cause of the 1x1 bug: det of a 0x0 matrix must be 1 (empty product)."""
        assert M([[1]]).determinant([]) == 1

    def test_determinant_explicit_argument_validated(self):
        """BUG-candidate: public `matrix=` argument is not validated (non-square silently 'works')."""
        with pytest.raises(ValueError):
            M([[1]]).determinant([[1, 2, 3], [4, 5, 6]])

    def test_determinant_optimised_ignores_matrix_argument(self):
        """BUG: signature advertises `matrix` but the argument is silently ignored."""
        A = M([[1, 2], [3, 4]])
        try:
            r = A.determinant_optimised([[5]])
        except TypeError:
            return                      # removing the dead parameter is also an acceptable fix
        assert r == pytest.approx(5)

    @pytest.mark.parametrize("rows", [[[0, 0], [0, 0]], [[1, 2], [2, 4]], [[1, 2], [1, 2]]],
                             ids=["zero", "proportional", "duplicate"])
    def test_singular_matrices(self, rows):
        A = M(rows)
        assert A.determinant() == 0 and A.determinant_optimised() == 0
        assert not A.is_invertible()
        for f in (A.inverse, A.inverse_optimised):
            with pytest.raises(ValueError):
                f()

    def test_3x3_known_values(self):
        A = M([[2, 0, 1], [1, 3, 2], [1, 1, 1]])
        assert A.determinant() == 2 * (3 - 2) - 0 + 1 * (1 - 3) == 0
        B = M([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        assert B.determinant() == 4 and B.determinant_optimised() == pytest.approx(4)
        assert B.trace() == 6 and B.is_symmetric()

    def test_row_swap_sign_in_pivoting_path(self):
        A = M([[0, 1], [1, 0]])
        assert A.determinant_optimised() == -1 == A.determinant()

    def test_permutation_and_reflection_are_orthogonal(self):
        assert M([[0, 1], [1, 0]]).is_orthogonal()
        assert M([[-1]]).is_orthogonal()
        assert not M([[1, 1], [0, 1]]).is_orthogonal()

    def test_inverse_of_zero_pivot_needing_swap(self):
        A = M([[0, 2], [3, 0]])
        for inv in (A.inverse(), A.inverse_optimised()):
            assert (A * inv).matrix == pytest.approx(np.eye(2))


# Projection
@pytest.mark.boundary
class TestProjectionBoundary:
    def test_project_onto_zero_vector(self):
        for f in (Projection.vector_projection, Projection.scalar_projection, Projection.rejection):
            with pytest.raises(ValueError):
                f(V(1, 2), V(0, 0))
        with pytest.raises(ValueError):
            Projection.projection_matrix(V(0, 0))

    def test_type_and_dimension_errors(self):
        with pytest.raises(TypeError):
            Projection.vector_projection([1, 2], V(1, 2))
        with pytest.raises(TypeError):
            Projection.vector_projection(V(1, 2), [1, 2])
        with pytest.raises(ValueError):
            Projection.vector_projection(V(1, 2), V(1, 2, 3))

    def test_projection_matrix_rejects_non_vector(self):
        """BUG-candidate: AttributeError instead of TypeError (no isinstance check)."""
        with pytest.raises(TypeError):
            Projection.projection_matrix([1, 2])

    def test_special_geometry(self):
        v = V(2, 0)
        assert Projection.vector_projection(V(2, 0), v) == V(2, 0)       # u == v
        assert Projection.vector_projection(V(0, 5), v) == V(0, 0)       # u ⟂ v
        assert Projection.vector_projection(V(0, 0), v) == V(0, 0)       # u == 0
        assert Projection.scalar_projection(V(-3, 1), v) == -3           # negative sign kept
        assert Projection.rejection(V(3, 4), v) == V(0, 4)

    @pytest.mark.parametrize("s", [1e-3, 1e-4, 1e-5, 1e-8])
    def test_small_but_nonzero_target_vector(self, s):
        """BUG: zero test is `v·v < 1e-9` (absolute) so any ||v|| < 3.2e-5 is declared 'zero'."""
        r = Projection.vector_projection(V(1, 1), V(s, 0))
        assert r == V(1.0, 0.0) or (r[0] == pytest.approx(1.0) and r[1] == 0)

    @pytest.mark.parametrize("s", [1e154, 1e160, 1e200])
    def test_huge_target_vector_does_not_silently_return_zero(self, s):
        """BUG: v·v overflows to inf, so the projection silently collapses to the zero vector."""
        r = Projection.vector_projection(V(1, 1), V(s, 0))
        assert r[0] == pytest.approx(1.0) and r[1] == 0


# GramSchmidt
@pytest.mark.boundary
class TestGramSchmidtBoundary:
    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            GramSchmidt.orthogonal_basis([])
        with pytest.raises(TypeError):
            GramSchmidt.orthogonal_basis([[1, 0]])
        with pytest.raises(TypeError):
            GramSchmidt.orthogonalize_vector([1, 0], [])
        with pytest.raises(ValueError):
            GramSchmidt.orthogonal_basis([V(0, 0), V(0, 0)])

    def test_dimension_mismatch_in_list(self):
        with pytest.raises(ValueError):
            GramSchmidt.orthogonal_basis([V(1, 0), V(0, 1, 0)])

    def test_dependent_duplicates_and_overfull_sets_are_dropped(self):
        assert len(GramSchmidt.orthogonal_basis([V(1, 1), V(2, 2), V(1, 0)])) == 2
        assert len(GramSchmidt.orthogonal_basis([V(1, 0), V(0, 1), V(1, 1), V(5, 7)])) == 2

    def test_zero_vector_in_middle_is_skipped(self):
        assert len(GramSchmidt.orthogonal_basis([V(1, 0, 0), V(0, 0, 0), V(0, 1, 0)])) == 2

    def test_orthogonalize_does_not_mutate_input(self):
        v = V(1, 1)
        GramSchmidt.orthogonalize_vector(v, [V(1, 0)])
        assert v == V(1, 1)

    def test_empty_basis_returns_copy(self):
        v = V(1, 2)
        out = GramSchmidt.orthogonalize_vector(v, [])
        assert out == v and out is not v

    def test_predicates_on_degenerate_inputs(self):
        assert GramSchmidt.is_orthogonal_basis([]) and GramSchmidt.is_orthogonal_basis([V(1, 2)])
        assert not GramSchmidt.is_orthonormal_basis([V(0, 0)])
        assert not GramSchmidt.is_orthonormal_basis([V(2, 0)])

    @pytest.mark.parametrize("s", [1e-3, 1e-6, 1e-9])
    def test_small_scale_input_is_not_discarded(self, s):
        """BUG: absolute 1e-9 thresholds treat small-but-independent vectors as 'dependent'/zero."""
        b = GramSchmidt.orthonormal_basis([V(s, 0), V(0, s)])
        assert len(b) == 2

    def test_verify_report_smoke(self, capsys):
        vs = [V(1, 0), V(1, 1)]
        GramSchmidt.verify(vs, GramSchmidt.orthonormal_basis(vs))
        assert "Valid orthonormal basis" in capsys.readouterr().out


# FundamentalSubspaces
@pytest.mark.boundary
class TestSubspacesBoundary:
    def test_zero_matrix(self):
        S = FundamentalSubspaces(LinearTransformation.zeros(2, 3))
        assert S.rank() == 0 and S.nullity() == 3
        assert S.column_space() == [] and S.row_space() == []
        assert len(S.null_space()) == 3 and len(S.left_null_space()) == 2

    def test_identity(self):
        S = FundamentalSubspaces(LinearTransformation.identity(4))
        assert S.rank() == 4 and S.nullity() == 0 and S.null_space() == []

    @pytest.mark.parametrize("rows,rank", [
        ([[0, 1], [0, 2]], 1),            # zero first column
        ([[0, 0], [1, 0]], 1),            # pivot not in first row
        ([[1, 2, 3]], 1), ([[1], [2], [3]], 1), ([[0]], 0), ([[-0.0, 0.0]], 0),
        ([[1, 2], [3, 4], [5, 6]], 2), ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2),
    ])
    def test_rank_edge_shapes(self, rows, rank):
        S = FundamentalSubspaces(M(rows))
        assert S.rank() == rank == np.linalg.matrix_rank(np.array(rows, float))
        assert S.rank() + S.nullity() == len(rows[0])

    def test_column_space_returns_original_columns_not_rref_columns(self):
        A = M([[0, 1, 2], [1, 0, 1]])
        assert FundamentalSubspaces(A).column_space() == [V(0, 1), V(1, 0)]

    def test_constructor_rejects_plain_lists(self):
        """BUG-candidate: AttributeError ('list' has no .shape) instead of a clean TypeError."""
        with pytest.raises(TypeError):
            FundamentalSubspaces([[1, 2], [3, 4]])

    @pytest.mark.parametrize("s", [1e-12, 1e-10, 1e-6])
    def test_rank_of_tiny_scaled_identity(self, s):
        """BUG: absolute pivot tolerance 1e-9: rank(s*I) == 0 for small s."""
        assert FundamentalSubspaces(LinearTransformation.identity(3) * s).rank() == 3

    def test_index_helpers_bounds(self):
        S = FundamentalSubspaces(M([[1, 2], [3, 4]]))
        for f in (S._extract_column, S._extract_row):
            with pytest.raises(IndexError):
                f(2)
            with pytest.raises(IndexError):
                f(-1)
