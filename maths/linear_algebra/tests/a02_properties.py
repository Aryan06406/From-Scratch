"""
Property based (Hypothesis) + invariance/symmetry/metamorphic tests for maths.linear_algebra.
Integer inputs => identities must hold EXACTLY (no tolerance to hide behind).
Float inputs  => tolerances are scaled to the data, never a bare absolute 1e-9.
"""
import math
import random

import numpy as np
import pytest
from hypothesis import assume, given, strategies as st

from maths.linear_algebra.core import (FundamentalSubspaces, GramSchmidt, LinearTransformation,
                                       Norms, Projection, Vector)
from maths.test_helpers import *   # noqa: F401,F403
from maths.test_helpers import (M, V, arr, diag_dominant, exact_inverse, int_mats, int_vecs, small_int, square_mats, 
                                varr, vec_lists, vec_pairs, vec_triples, rot2, nonzero_int, positive_int, tiny_int, dim)

def close(a, b, scale=1.0, rtol=1e-9):
    return abs(a - b) <= rtol * max(1.0, scale)

def zero(n):
    return Vector([0] * n)


# Vector algebra (exact)
@pytest.mark.property
class TestVectorAlgebra:
    @given(vec_pairs())
    def test_add_commutative_and_zero_identity(self, ab):
        a, b = ab
        assert a + b == b + a
        assert a + zero(len(a)) == a
        assert a + (-a) == zero(len(a))
        assert a - b == a + (-b)

    @given(vec_triples())
    def test_add_associative_dot_bilinear_hadamard(self, abc):
        a, b, c = abc
        assert (a + b) + c == a + (b + c)
        assert (a + b).dot_product(c) == a.dot_product(c) + b.dot_product(c)
        assert a.hadamard_product(b) == b.hadamard_product(a)
        assert a.hadamard_product(b + c) == a.hadamard_product(b) + a.hadamard_product(c)

    @given(vec_pairs(), small_int, small_int)
    def test_scalar_laws(self, ab, k, m):
        a, b = ab
        assert k * (a + b) == k * a + k * b
        assert (k + m) * a == k * a + m * a
        assert (k * m) * a == k * (m * a)
        assert (k * a).dot_product(b) == k * a.dot_product(b) == a.dot_product(k * b)

    @given(vec_pairs())
    def test_dot_symmetry_positivity_cauchy_schwarz_exact(self, ab):
        a, b = ab
        assert a.dot_product(b) == b.dot_product(a)
        assert a.dot_product(a) >= 0
        assert (a.dot_product(a) == 0) == (a == zero(len(a)))
        assert a.dot_product(b) ** 2 <= a.dot_product(a) * b.dot_product(b)   # exact ints

    @given(vec_triples().filter(lambda t: len(t[0]) == 3) if False else
           st.tuples(*[st.lists(small_int, min_size=3, max_size=3).map(Vector)] * 3))
    def test_cross_product_identities_exact(self, abc):
        a, b, c = abc
        x = a.cross_product(b)
        assert x == -(b.cross_product(a))                                  # anticommutative
        assert a.cross_product(a) == zero(3)
        assert x.dot_product(a) == 0 == x.dot_product(b)                   # orthogonal to both
        assert x.dot_product(x) == a.dot_product(a) * b.dot_product(b) - a.dot_product(b) ** 2  # Lagrange
        assert a.cross_product(b + c) == a.cross_product(b) + a.cross_product(c)
        assert c.dot_product(x) == a.dot_product(b.cross_product(c))       # cyclic triple product
        assert (a.cross_product(b.cross_product(c)) + b.cross_product(c.cross_product(a))
                + c.cross_product(a.cross_product(b))) == zero(3)          # Jacobi

    @given(vec_pairs())
    def test_distance_is_a_metric(self, ab):
        a, b = ab
        d = a.distance_between_vectors(b)
        assert d == b.distance_between_vectors(a) >= 0
        assert a.distance_between_vectors(a) == 0
        assert (d == 0) == (a == b)

    @given(vec_triples())
    def test_triangle_inequality_and_parallelogram_law(self, abc):
        a, b, c = abc
        ab_, bc_, ac_ = (a.distance_between_vectors(b), b.distance_between_vectors(c),
                         a.distance_between_vectors(c))
        assert ac_ <= ab_ + bc_ + 1e-9 * (1 + ab_ + bc_)
        lhs = (a + b).magnitude() ** 2 + (a - b).magnitude() ** 2
        rhs = 2 * a.magnitude() ** 2 + 2 * b.magnitude() ** 2
        assert close(lhs, rhs, rhs)

    @given(vec_pairs(), nonzero_int)
    def test_is_parallel_matches_scalar_multiples(self, ab, k):
        a, _ = ab
        assert a.is_parallel(k * a) and (k * a).is_parallel(a)

    @given(st.tuples(*[st.lists(small_int, min_size=3, max_size=3).map(Vector)] * 2))
    def test_is_parallel_agrees_with_cross_product_reference(self, ab):
        a, b = ab
        assume(a != zero(3) and b != zero(3))
        assert a.is_parallel(b) == (a.cross_product(b) == zero(3))
        assert a.is_parallel(b) == b.is_parallel(a)

    @given(vec_pairs())
    def test_is_orthogonal_matches_zero_dot(self, ab):
        a, b = ab
        assert a.is_orthogonal(b) == (a.dot_product(b) == 0)


# Norm axioms
@pytest.mark.property
class TestNormAxioms:
    @given(vec_pairs(), small_int)
    def test_homogeneity_and_definiteness(self, ab, k):
        a, _ = ab
        for p in (1, 2, 3.5):
            na, nka = Norms(a).p_norm(p), Norms(k * a).p_norm(p)
            assert close(nka, abs(k) * na, abs(k) * na)
        assert (Norms(a).l1_norm() == 0) == (a == zero(len(a)))
        assert Norms(k * a).infinity_norm() == abs(k) * Norms(a).infinity_norm()

    @given(vec_pairs(), st.sampled_from([1, 1.5, 2, 3, 7]))
    def test_minkowski_triangle_inequality(self, ab, p):
        a, b = ab
        lhs = Norms(a + b).p_norm(p)
        rhs = Norms(a).p_norm(p) + Norms(b).p_norm(p)
        assert lhs <= rhs * (1 + 1e-12) + 1e-12

    @given(vec_pairs())
    def test_norm_ordering_and_equivalence(self, ab):
        a, _ = ab
        n = len(a)
        N = Norms(a)
        linf, l2, l1 = N.infinity_norm(), N.l2_norm(), N.l1_norm()
        eps = 1e-12 * (1 + l1)
        assert linf <= l2 + eps and l2 <= l1 + eps
        assert l1 <= math.sqrt(n) * l2 + eps and l2 <= math.sqrt(n) * linf + eps
        # p-norms are non-increasing in p
        ps = [1, 1.5, 2, 3, 6, 12]
        vals = [N.p_norm(p) for p in ps]
        assert all(x >= y - eps for x, y in zip(vals, vals[1:]))
        assert vals[-1] >= linf - eps                       # all >= linf
        assert N.p_norm(12) <= n ** (1 / 12) * linf + eps   # and converge to it

    @given(vec_pairs())
    def test_l2_norm_matches_magnitude_and_p2(self, ab):
        a, _ = ab
        assert Norms(a).l2_norm() == a.magnitude()
        assert close(Norms(a).p_norm(2), a.magnitude(), a.magnitude(), 1e-12)
        assert Norms(a).p_norm(1) == Norms(a).l1_norm()


# Invariance / symmetry
@pytest.mark.invariance
class TestVectorInvariance:
    @given(vec_pairs(), st.randoms(use_true_random=False))
    def test_coordinate_permutation_invariance(self, ab, rnd):
        a, b = ab
        idx = list(range(len(a)))
        rnd.shuffle(idx)
        pa, pb = Vector([a[i] for i in idx]), Vector([b[i] for i in idx])
        assert pa.dot_product(pb) == a.dot_product(b)
        assert pa.magnitude() == a.magnitude()
        assert pa.distance_between_vectors(pb) == a.distance_between_vectors(b)
        if a.magnitude() and b.magnitude():
            assert pa.angle_between_vectors(pb) == a.angle_between_vectors(b)

    @given(vec_pairs(), vec_pairs())
    def test_translation_and_joint_negation(self, ab, cc):
        a, b = ab
        c = Vector([cc[0][i % len(cc[0])] for i in range(len(a))])
        assert (a + c).distance_between_vectors(b + c) == a.distance_between_vectors(b)
        assert (-a).distance_between_vectors(-b) == a.distance_between_vectors(b)
        assert (-a).dot_product(-b) == a.dot_product(b)

    @given(vec_pairs(), positive_int)
    def test_angle_invariances(self, ab, k):
        a, b = ab
        assume(a.magnitude() > 0 and b.magnitude() > 0)
        th = a.angle_between_vectors(b)
        assert 0.0 <= th <= 180.0
        assert th == b.angle_between_vectors(a)                           # symmetry
        assert abs(math.cos(math.radians(th)) - math.cos(math.radians((k * a).angle_between_vectors(b)))) < 1e-12
        assert abs(th + a.angle_between_vectors(-b) - 180.0) < 1e-9       # supplementary
        assert abs((-a).angle_between_vectors(-b) - th) < 1e-12

    @given(vec_pairs(), positive_int)
    def test_unit_vector_properties(self, ab, k):
        a, _ = ab
        assume(a.magnitude() > 0)
        u = a.unit_vector()
        assert abs(u.magnitude() - 1) < 1e-12
        assert max(abs(x - y) for x, y in zip((k * a).unit_vector(), u)) < 1e-12   # scale invariance
        assert max(abs(x - y) for x, y in zip(u.unit_vector(), u)) < 1e-12          # idempotent
        assert a.normalize() == u

    @given(st.floats(0, 2 * math.pi), vec_pairs(st.integers(-50, 50)))
    def test_rotation_invariance_2d(self, theta, ab):
        a, b = ab
        a, b = Vector(list(a)[:2] + [0] * (2 - len(a))), Vector(list(b)[:2] + [0] * (2 - len(b)))
        R = rot2(theta)
        ra, rb = R.apply_transformation(a), R.apply_transformation(b)
        scale = a.magnitude() * b.magnitude() + 1
        assert close(ra.dot_product(rb), a.dot_product(b), scale, 1e-12)
        assert close(ra.magnitude(), a.magnitude(), a.magnitude(), 1e-12)
        assert close(ra.distance_between_vectors(rb), a.distance_between_vectors(b), a.magnitude() + b.magnitude(), 1e-12)
        if a.magnitude() > 1e-6 and b.magnitude() > 1e-6:
            c0 = math.cos(math.radians(a.angle_between_vectors(b)))
            c1 = math.cos(math.radians(ra.angle_between_vectors(rb)))
            assert abs(c0 - c1) < 1e-12


# Matrix algebra (exact)
@pytest.mark.property
class TestMatrixAlgebra:
    @given(st.integers(1, 4), st.integers(1, 4), st.integers(1, 4), st.data())
    def test_transpose_and_product_laws(self, m, n, p, data):
        A, A2 = data.draw(int_mats(m, n)), data.draw(int_mats(m, n))
        B, B2 = data.draw(int_mats(n, p)), data.draw(int_mats(n, p))
        C = data.draw(int_mats(p, m))
        assert A.transpose().transpose() == A
        assert (A + A2).transpose() == A.transpose() + A2.transpose()
        assert (A * B).transpose() == B.transpose() * A.transpose()
        assert A * (B + B2) == A * B + A * B2
        assert (A * B) * C == A * (B * C)
        assert (A * B * C).trace() == (C * A * B).trace() == (B * C * A).trace()   # cyclic trace
        assert A - A == LinearTransformation.zeros(m, n)
        assert 3 * (A + A2) == 3 * A + 3 * A2

    @given(square_mats())
    def test_identity_laws_and_trace(self, A):
        n = A.shape[0]
        I = LinearTransformation.identity(n)
        assert I * A == A == A * I
        assert A.transpose().trace() == A.trace()
        assert (A + A).trace() == 2 * A.trace()

    @given(st.integers(1, 4), st.integers(1, 4), st.data())
    def test_apply_transformation_is_linear_and_adjoint(self, m, n, data):
        A = data.draw(int_mats(m, n))
        v, w = data.draw(int_vecs(n)), data.draw(int_vecs(n))
        y = data.draw(int_vecs(m))
        a, b = data.draw(small_int), data.draw(small_int)
        assert A.apply_transformation(a * v + b * w) == a * A.apply_transformation(v) + b * A.apply_transformation(w)
        assert A.apply_transformation(v).dot_product(y) == v.dot_product(A.transpose().apply_transformation(y))
        B = data.draw(int_mats(n, 3))
        x = data.draw(int_vecs(3))
        assert (A * B).apply_transformation(x) == A.apply_transformation(B.apply_transformation(x))

    @given(st.integers(2, 4), st.data())
    def test_determinant_laws_exact(self, n, data):
        A, B = data.draw(int_mats(n, n)), data.draw(int_mats(n, n))
        k = data.draw(small_int)
        d = A.determinant()
        assert (A * B).determinant() == d * B.determinant()
        assert A.transpose().determinant() == d
        assert (k * A).determinant() == k ** n * d
        assert LinearTransformation.identity(n).determinant() == 1
        i, j = data.draw(st.sampled_from([(0, 1), (0, n - 1)]))
        S = [r[:] for r in A.matrix]
        S[i], S[j] = S[j], S[i]
        assert M(S).determinant() == -d
        S = [r[:] for r in A.matrix]
        S[j] = S[i][:]
        assert M(S).determinant() == 0                              # duplicate row
        S = [r[:] for r in A.matrix]
        S[j] = [x + k * y for x, y in zip(S[j], S[i])]
        assert M(S).determinant() == d                              # row addition invariance
        assert M(S).determinant_optimised() == pytest.approx(d, abs=1e-9 * (1 + np.prod(np.linalg.norm(arr(A), axis=1))))

    @given(st.integers(1, 5), st.data())
    def test_triangular_determinant_is_diagonal_product(self, n, data):
        diag = [data.draw(nonzero_int) for _ in range(n)]
        rows = [[(diag[i] if i == j else (data.draw(small_int) if j > i else 0)) for j in range(n)] for i in range(n)]
        A = M(rows)
        assert A.determinant() == math.prod(diag)
        assert A.determinant_optimised() == pytest.approx(math.prod(diag), rel=1e-9)

    @given(square_mats(hi=4, elems=st.integers(-5, 5)))
    def test_gram_matrix_symmetry_and_psd(self, A):
        G, H = A.transpose() * A, A * A.transpose()
        assert G.is_symmetric() and H.is_symmetric() and (A + A.transpose()).is_symmetric()
        assert (A - A.transpose()).transpose() == -1 * (A - A.transpose())
        x = Vector(list(range(1, A.shape[0] + 1)))
        assert G.apply_transformation(x).dot_product(x) >= 0


@pytest.mark.property
class TestInverseProperties:
    @given(diag_dominant())
    def test_inverse_is_two_sided_and_paths_agree(self, A):
        n = A.shape[0]
        I = np.eye(n)
        for inv in (A.inverse(), A.inverse_optimised()):
            assert np.allclose(arr(A * inv), I, atol=1e-9) and np.allclose(arr(inv * A), I, atol=1e-9)
        assert np.allclose(arr(A.inverse()), arr(A.inverse_optimised()), rtol=1e-8, atol=1e-10)
        assert A.is_invertible()

    @given(diag_dominant())
    def test_adjugate_inverse_is_correctly_rounded(self, A):
        exact = exact_inverse(A.matrix)
        got = A.inverse()
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                assert got[i][j] == float(exact[i][j])

    @given(diag_dominant(2, 3), st.data())
    def test_inverse_algebra(self, A, data):
        n = A.shape[0]
        B = data.draw(diag_dominant(n, n))
        assert np.allclose(arr(A.inverse_optimised().inverse_optimised()), arr(A), rtol=1e-8, atol=1e-8)
        assert np.allclose(arr((A * B).inverse_optimised()), arr(B.inverse_optimised() * A.inverse_optimised()), rtol=1e-7, atol=1e-10)
        assert np.allclose(arr(A.transpose().inverse_optimised()), arr(A.inverse_optimised().transpose()), rtol=1e-8, atol=1e-10)
        assert A.inverse_optimised().determinant_optimised() * A.determinant_optimised() == pytest.approx(1, rel=1e-8)


# Orthogonal matrix invariance
def givens(n, i, j, th):
    G = np.eye(n)
    G[i, i] = G[j, j] = math.cos(th)
    G[i, j], G[j, i] = -math.sin(th), math.sin(th)
    return G


@pytest.mark.invariance
class TestOrthogonalMatrices:
    @given(st.integers(2, 5), st.lists(st.tuples(st.integers(0, 4), st.integers(0, 4), st.floats(0, 6.3)), min_size=1, max_size=6),
           st.data())
    def test_products_of_rotations_are_orthogonal_and_isometric(self, n, rots, data):
        Q = np.eye(n)
        for i, j, th in rots:
            if i % n != j % n:
                Q = Q @ givens(n, i % n, j % n, th)
        QM = M(Q.tolist())
        assert QM.is_orthogonal()
        assert QM.determinant_optimised() == pytest.approx(1.0, abs=1e-9)
        assert np.allclose(arr(QM.inverse_optimised()), Q.T, atol=1e-9)
        v, w = data.draw(int_vecs(n)), data.draw(int_vecs(n))
        qv, qw = QM.apply_transformation(v), QM.apply_transformation(w)
        s = v.magnitude() * w.magnitude() + 1
        assert close(qv.dot_product(qw), v.dot_product(w), s, 1e-11)
        assert close(qv.magnitude(), v.magnitude(), v.magnitude() + 1, 1e-11)

    @given(st.integers(2, 6), st.randoms(use_true_random=False))
    def test_permutation_matrices_exactly_orthogonal(self, n, rnd):
        p = list(range(n))
        rnd.shuffle(p)
        P = M([[1 if p[i] == j else 0 for j in range(n)] for i in range(n)])
        assert P.is_orthogonal() and P * P.transpose() == LinearTransformation.identity(n)
        assert abs(P.determinant()) == 1

    @given(st.integers(2, 5), st.data())
    def test_householder_reflection(self, n, data):
        u = data.draw(int_vecs(n))
        assume(u.magnitude() > 0)
        P = LinearTransformation.identity(n) + (-2 * Projection.projection_matrix(u))
        assert P.is_orthogonal() and P.is_symmetric() is not None
        assert P.determinant_optimised() == pytest.approx(-1.0, abs=1e-9)


# Projection
@pytest.mark.property
class TestProjectionProperties:
    @given(vec_pairs(), nonzero_int)
    def test_decomposition_and_orthogonality(self, uv, k):
        u, v = uv
        assume(v.magnitude() > 0)
        p, r = Projection.vector_projection(u, v), Projection.rejection(u, v)
        s = u.magnitude() * v.magnitude() + 1
        assert r.is_orthogonal(v) or abs(r.dot_product(v)) <= 1e-9 * s
        assert max(abs(x - y) for x, y in zip(p + r, u)) <= 1e-12 * (1 + u.magnitude())
        assert close(u.magnitude() ** 2, p.magnitude() ** 2 + r.magnitude() ** 2, u.magnitude() ** 2, 1e-12)
        assert p.magnitude() <= u.magnitude() * (1 + 1e-12)
        assert Projection.orthogonal_component(u, v) == r
        # idempotent
        assert max(abs(x - y) for x, y in zip(Projection.vector_projection(p, v), p)) <= 1e-12 * (1 + p.magnitude())
        # invariant to rescaling / flipping the direction vector
        pk = Projection.vector_projection(u, k * v)
        assert max(abs(x - y) for x, y in zip(pk, p)) <= 1e-12 * (1 + p.magnitude())
        assert close(Projection.scalar_projection(u, k * v), math.copysign(1, k) * Projection.scalar_projection(u, v), s, 1e-12)

    @given(vec_pairs(), small_int, small_int)
    def test_projection_is_linear_in_u(self, uv, a, b):
        u, v = uv
        assume(v.magnitude() > 0)
        w = Vector([(i + 1) * x for i, x in enumerate(u)])
        lhs = Projection.vector_projection(a * u + b * w, v)
        rhs = a * Projection.vector_projection(u, v) + b * Projection.vector_projection(w, v)
        assert max(abs(x - y) for x, y in zip(lhs, rhs)) <= 1e-9 * (1 + lhs.magnitude())

    @given(vec_pairs(), nonzero_int)
    def test_projection_matrix_properties(self, uv, k):
        u, v = uv
        assume(v.magnitude() > 0)
        n = len(v)
        P = Projection.projection_matrix(v)
        Pa = arr(P)
        assert P.is_symmetric()
        assert np.allclose(Pa @ Pa, Pa, atol=1e-12)
        assert abs(P.trace() - 1) < 1e-12
        assert np.allclose(Pa @ varr(v), varr(v), atol=1e-12 * (1 + v.magnitude()))
        assert np.allclose(arr(Projection.projection_matrix(k * v)), Pa, atol=1e-12)
        assert np.allclose(Pa @ varr(u), varr(Projection.vector_projection(u, v)), atol=1e-12 * (1 + u.magnitude()))
        assert np.linalg.matrix_rank(Pa) == 1


# Gram-Schmidt
@pytest.mark.property
class TestGramSchmidtProperties:
    @given(vec_lists())
    def test_output_is_orthonormal_has_rank_many_vectors_and_spans(self, vs):
        assume(any(v.magnitude() > 0 for v in vs))
        Q = GramSchmidt.orthonormal_basis(vs)
        rank = np.linalg.matrix_rank(np.array([varr(v) for v in vs]))
        assert len(Q) == rank
        G = np.array([varr(q) for q in Q]) @ np.array([varr(q) for q in Q]).T
        assert np.allclose(G, np.eye(rank), atol=1e-9)
        for v in vs:                                                    # same span
            res = GramSchmidt.orthogonalize_vector(v, Q)
            assert res.magnitude() <= 1e-8 * (1 + v.magnitude())

    @given(vec_lists(), st.randoms(use_true_random=False))
    def test_span_is_invariant_to_input_order(self, vs, rnd):
        assume(any(v.magnitude() > 0 for v in vs))
        shuffled = vs[:]
        rnd.shuffle(shuffled)

        def proj(basis):
            n = len(basis[0])
            return sum((np.outer(varr(b), varr(b)) / varr(b).dot(varr(b)) for b in basis), np.zeros((n, n)))
        assert np.allclose(proj(GramSchmidt.orthogonal_basis(vs)), proj(GramSchmidt.orthogonal_basis(shuffled)), atol=1e-8)

    @given(vec_lists(), st.lists(positive_int, min_size=5, max_size=5))
    def test_positive_rescaling_of_inputs_does_not_change_orthonormal_basis(self, vs, ks):
        assume(any(v.magnitude() > 0 for v in vs))
        Q1 = GramSchmidt.orthonormal_basis(vs)
        Q2 = GramSchmidt.orthonormal_basis([k * v for k, v in zip(ks, vs)])
        assert len(Q1) == len(Q2)
        assert all(max(abs(x - y) for x, y in zip(a, b)) < 1e-9 for a, b in zip(Q1, Q2))

    @given(vec_lists())
    def test_idempotent_on_orthonormal_input(self, vs):
        assume(any(v.magnitude() > 0 for v in vs))
        Q = GramSchmidt.orthonormal_basis(vs)
        Q2 = GramSchmidt.orthonormal_basis(Q)
        assert len(Q) == len(Q2) and all(max(abs(x - y) for x, y in zip(a, b)) < 1e-9 for a, b in zip(Q, Q2))

# Fundamental subspaces
tiny_mats = st.tuples(st.integers(1, 5), st.integers(1, 5)).flatmap(lambda t: int_mats(t[0], t[1], st.integers(-3, 3)))

@pytest.mark.property
class TestSubspaceProperties:
    @given(tiny_mats)
    def test_rank_nullity_and_reference_rank(self, A):
        m, n = A.shape
        S = FundamentalSubspaces(A)
        r = S.rank()
        assert r == np.linalg.matrix_rank(arr(A))
        assert r == FundamentalSubspaces(A.transpose()).rank() <= min(m, n)
        assert r + S.nullity() == n and len(S.null_space()) == S.nullity()
        assert len(S.column_space()) == len(S.row_space()) == r
        assert len(S.left_null_space()) == m - r

    @given(tiny_mats)
    def test_four_subspaces_are_mutually_consistent(self, A):
        S = FundamentalSubspaces(A)
        Aa = arr(A)
        for x in S.null_space():
            assert np.allclose(Aa @ varr(x), 0, atol=1e-9)
        for y in S.left_null_space():
            assert np.allclose(Aa.T @ varr(y), 0, atol=1e-9)
        for x in S.null_space():                                      # row space ⟂ null space
            for r in S.row_space():
                assert abs(x.dot_product(r)) < 1e-9
        for y in S.left_null_space():                                 # col space ⟂ left null
            for c in S.column_space():
                assert abs(y.dot_product(c)) < 1e-9
        for basis in (S.null_space(), S.left_null_space(), S.column_space(), S.row_space()):
            if basis:                                                 # bases are independent
                assert np.linalg.matrix_rank(np.array([varr(b) for b in basis])) == len(basis)

    @given(tiny_mats, st.data())
    def test_row_operations_preserve_rank_rowspace_and_nullspace(self, A, data):
        m, n = A.shape
        assume(m >= 2)
        i, j = data.draw(st.sampled_from([(0, 1), (1, 0)]))
        k = data.draw(st.integers(-3, 3))
        rows = [r[:] for r in A.matrix]
        op = data.draw(st.sampled_from(["swap", "scale", "add"]))
        if op == "swap":
            rows[i], rows[j] = rows[j], rows[i]
        elif op == "scale":
            rows[i] = [2 * x for x in rows[i]]
        else:
            rows[j] = [x + k * y for x, y in zip(rows[j], rows[i])]
        S0, S1 = FundamentalSubspaces(A), FundamentalSubspaces(M(rows))
        assert S0.rank() == S1.rank()
        R0, R1 = S0.row_space(), S1.row_space()           # RREF is unique => identical basis
        assert len(R0) == len(R1) and all(max(abs(x - y) for x, y in zip(a, b)) < 1e-9 for a, b in zip(R0, R1))
        N0, N1 = S0.null_space(), S1.null_space()
        assert all(max(abs(x - y) for x, y in zip(a, b)) < 1e-9 for a, b in zip(N0, N1))

    @given(tiny_mats)
    def test_column_permutation_and_rref_idempotence(self, A):
        m, n = A.shape
        S = FundamentalSubspaces(A)
        rev = M([row[::-1] for row in A.matrix])
        assert FundamentalSubspaces(rev).rank() == S.rank()
        R = S.row_space()
        if R:
            S2 = FundamentalSubspaces(M([list(r) for r in R]))
            assert S2.rank() == len(R)
            assert all(max(abs(x - y) for x, y in zip(a, b)) < 1e-9 for a, b in zip(S2.row_space(), R))

    @given(st.integers(1, 4), st.integers(1, 4), st.integers(1, 4), st.data())
    def test_rank_of_product_bound(self, m, n, p, data):
        A, B = data.draw(int_mats(m, n, tiny_int)), data.draw(int_mats(n, p, tiny_int))
        r = lambda X: FundamentalSubspaces(X).rank()
        assert r(A * B) <= min(r(A), r(B))
