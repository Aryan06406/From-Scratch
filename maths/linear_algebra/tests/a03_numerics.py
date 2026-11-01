"""
Numerical-precision / tolerance / reference-implementation / stress tests for maths.linear_algebra.

Reference oracles:  numpy.linalg (LAPACK), math.hypot / math.fsum, exact Fractions.
Tolerances are derived from floating-point error theory (n*u*sum|a_i b_i|, cond(A)*eps),
NOT from "whatever makes the test pass".
"""
import math

import numpy as np
import pytest
from fractions import Fraction
from hypothesis import assume, given, strategies as st

from maths.linear_algebra.core import (FundamentalSubspaces, GramSchmidt, LinearTransformation,
                                       Norms, Projection, Vector)
from maths.test_helpers import (EPS, U, M, V, arr, exact_det, exact_inverse, hilbert,
                                             mat, vec, varr, rot2)

rng = np.random.default_rng(20261024)


@pytest.fixture(autouse=True)
def _reseed():
    """Deterministic data per test, independent of test selection / ordering."""
    global rng
    rng = np.random.default_rng(20261024)


# ================================================================ Vector precision
@pytest.mark.numeric
class TestVectorPrecision:
    @given(st.lists(st.tuples(st.floats(-1e6, 1e6), st.floats(-1e6, 1e6)), min_size=1, max_size=50))
    def test_dot_product_meets_higham_forward_error_bound(self, pairs):
        a, b = Vector([p[0] for p in pairs]), Vector([p[1] for p in pairs])
        exact = math.fsum(x * y for x, y in pairs)          # (x*y rounding only; error bound still holds up to 2u)
        bound = (len(pairs) + 2) * U * math.fsum(abs(x * y) for x, y in pairs) * 1.0001
        assert abs(a.dot_product(b) - exact) <= bound

    @pytest.mark.parametrize("scale", [1e-200, 1e-160, 1e-100, 1.0, 1e100, 1e154, 1e160, 1e200])
    def test_magnitude_vs_hypot_across_exponent_range(self, scale):
        """BUG: x**2 overflows >1e154 (OverflowError) and underflows <1e-162 (returns 0.0)."""
        xs = [3 * scale, 4 * scale, 12 * scale]
        assert Vector(xs).magnitude() == pytest.approx(math.hypot(*xs), rel=4 * EPS)

    @pytest.mark.parametrize("scale", [1e-200, 1e-170, 1e170, 1e200])
    def test_unit_vector_and_angle_at_extreme_scales(self, scale):
        v = V(scale, 2 * scale)
        assert v.unit_vector().magnitude() == pytest.approx(1.0, rel=1e-12)
        assert v.angle_between_vectors(V(1.0, 2.0)) == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("eps_angle", [1e-4, 1e-6, 1e-8, 1e-10])
    def test_small_angle_resolution(self, eps_angle):
        """BUG: acos(dot/|a||b|) loses everything below ~1e-8 rad (cos rounds to 1.0)."""
        got = V(1, 0).angle_between_vectors(V(1, eps_angle))
        ref = math.degrees(math.atan2(eps_angle, 1.0))
        assert got == pytest.approx(ref, rel=1e-6)

    @pytest.mark.parametrize("eps_angle", [1e-6, 1e-8])
    def test_near_antiparallel_angle_resolution(self, eps_angle):
        got = V(-1, 0).angle_between_vectors(V(1, eps_angle))
        ref = 180.0 - math.degrees(math.atan2(eps_angle, 1.0))
        assert 180.0 - got == pytest.approx(180.0 - ref, rel=1e-6)

    @given(st.lists(st.floats(-1e8, 1e8), min_size=2, max_size=6))
    def test_angle_with_self_never_raises_and_is_tiny(self, xs):
        v = Vector(xs)
        assume(v.magnitude() > 1e-100)
        assert abs(v.angle_between_vectors(v)) < 1e-5            # acos clamp must prevent domain errors

    @pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9])
    def test_is_orthogonal_decision_is_scale_invariant(self, scale):
        """BUG: `abs(dot) < 1e-9` is absolute. Tiny vectors look orthogonal; huge ones never are."""
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        a, b_orth = vec(scale * Q[:, 0]), vec(scale * Q[:, 1])           # orthogonal to ~1e-16 relative
        b_tilt = vec(scale * (np.cos(1.4) * Q[:, 0] + np.sin(1.4) * Q[:, 1]))   # 80 degrees from a
        assert a.is_orthogonal(b_orth), "orthogonal pair rejected"
        assert not a.is_orthogonal(b_tilt), "80-degree pair accepted as orthogonal"

    @pytest.mark.parametrize("scale", [1.0, 1e3, 1e8])
    def test_is_parallel_tolerates_rounding(self, scale):
        """BUG: exact float equality (`a*r1 != b*r0`) rejects mathematically parallel float vectors."""
        a = Vector([0.1 * scale, 0.2 * scale, 0.3 * scale])
        assert a.is_parallel(a * 3.0) and a.is_parallel(a * 0.7) and a.is_parallel(a / 3.0)

    @pytest.mark.parametrize("p", [3, 20, 100, 1000])
    @pytest.mark.parametrize("scale", [1.0, 1e10, 1e-10])
    def test_p_norm_is_overflow_safe(self, p, scale):
        """BUG: |x|**p overflows (OverflowError) / underflows (0.0); needs max-scaling."""
        xs = [1.0 * scale, 2.0 * scale, 3.0 * scale]
        m = max(xs)
        ref = m * sum((x / m) ** p for x in xs) ** (1 / p)
        assert Norms(Vector(xs)).p_norm(p) == pytest.approx(ref, rel=1e-12)

    def test_p_norm_converges_to_infinity_norm(self):
        v = V(1.0, -7.0, 3.0)
        got = Norms(v).p_norm(5000)
        assert got == pytest.approx(7.0, rel=1e-3)

    def test_cross_product_catastrophic_cancellation_is_bounded(self):
        a, b = V(1e8, 1.0, 0.0), V(1e8 + 1.0, 1.0 + 1e-8, 0.0)
        z = a.cross_product(b)[2]
        exact = float(Fraction(1e8) * Fraction(1.0 + 1e-8) - Fraction(1.0) * Fraction(1e8 + 1.0))
        assert abs(z - exact) <= 4 * U * (abs(a[0] * b[1]) + abs(a[1] * b[0]))


# ================================================================ Matrix vs references
def random_well_conditioned(n, seed):
    r = np.random.default_rng(seed)
    return r.standard_normal((n, n)) + n * np.eye(n) * 0.5


@pytest.mark.reference
class TestMatrixAgainstNumpy:
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_determinants_against_lapack(self, n):
        for seed in range(10):
            A = rng.standard_normal((n, n))
            had = float(np.prod(np.linalg.norm(A, axis=1)))               # Hadamard bound
            ref = np.linalg.det(A)
            tol = 50 * n * EPS * had
            assert M(A.tolist()).determinant_optimised() == pytest.approx(ref, abs=tol)
            assert M(A.tolist()).determinant() == pytest.approx(ref, abs=tol)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_inverses_against_lapack_with_condition_scaled_tolerance(self, n):
        for seed in range(10):
            A = rng.standard_normal((n, n))
            kappa = np.linalg.cond(A)
            ref = np.linalg.inv(A)
            for inv in (M(A.tolist()).inverse_optimised(),) + ((M(A.tolist()).inverse(),) if n <= 6 else ()):
                err = np.linalg.norm(arr(inv) - ref) / np.linalg.norm(ref)
                assert err <= 100 * n * EPS * kappa, f"rel err {err:.2e}, cond {kappa:.2e}"
                resid = np.linalg.norm(A @ arr(inv) - np.eye(n))
                assert resid <= 100 * n * EPS * kappa

    @pytest.mark.parametrize("shape", [(3, 4, 5), (7, 1, 6), (1, 9, 1), (20, 20, 20)])
    def test_matmul_and_matvec_against_numpy(self, shape):
        m, k, n = shape
        A, B, x = rng.standard_normal((m, k)), rng.standard_normal((k, n)), rng.standard_normal(k)
        assert np.allclose(arr(mat(A) * mat(B)), A @ B, rtol=1e-12, atol=1e-12 * k)
        assert np.allclose(varr(mat(A).apply_transformation(vec(x))), A @ x, rtol=1e-12, atol=1e-12 * k)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_integer_matrices_exact_against_fractions(self, n):
        for _ in range(15):
            A = rng.integers(-9, 10, size=(n, n)).tolist()
            d = exact_det(A)
            assert M(A).determinant() == d                                   # cofactor path: exact ints
            assert M(A).determinant_optimised() == pytest.approx(float(d), abs=1e-9 * (1 + abs(float(d))))
            if d != 0 and n >= 2:
                ex = exact_inverse(A)
                got = M(A).inverse()
                assert all(got[i][j] == float(ex[i][j]) for i in range(n) for j in range(n))

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_rank_against_numpy_on_random_low_rank_float_matrices(self, n):
        for r in range(0, n + 1):
            A = rng.standard_normal((n, r)) @ rng.standard_normal((r, n)) if r else np.zeros((n, n))
            assert FundamentalSubspaces(mat(A)).rank() == np.linalg.matrix_rank(A) == r

    @pytest.mark.parametrize("scale", [1e-12, 1e-6, 1.0, 1e6, 1e12])
    def test_rank_decision_is_scale_invariant(self, scale):
        """BUG: absolute 1e-9 pivot tolerance => rank(c*A) depends on c (0 for tiny c, full for huge c)."""
        A = scale * (rng.standard_normal((6, 3)) @ rng.standard_normal((3, 6)))
        assert FundamentalSubspaces(mat(A)).rank() == 3

    @pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6, 1e9])
    def test_null_space_residual_relative_to_matrix_norm(self, scale):
        A = scale * (rng.standard_normal((7, 4)) @ rng.standard_normal((4, 7)))
        S = FundamentalSubspaces(mat(A))
        assert len(S.null_space()) == 3
        for x in S.null_space():
            assert np.linalg.norm(A @ varr(x)) <= 1e-9 * np.linalg.norm(A) * np.linalg.norm(varr(x))
        for y in S.left_null_space():
            assert np.linalg.norm(A.T @ varr(y)) <= 1e-9 * np.linalg.norm(A) * np.linalg.norm(varr(y))

    @pytest.mark.parametrize("scale", [1e-4, 1e-3, 1e-2, 1.0, 1e3])
    def test_is_invertible_decision_is_scale_invariant(self, scale):
        """BUG: `abs(det) > 1e-9` => 1e-4*I(3) (perfectly conditioned) is 'singular'."""
        assert (LinearTransformation.identity(3) * scale).is_invertible()

    @pytest.mark.parametrize("scale", [1e3, 1e6, 1e9])
    def test_is_invertible_rejects_float_singular_matrices_at_any_scale(self, scale):
        """BUG: det of a float rank-1 matrix is rounding noise ~ scale^2*1e-16 which exceeds 1e-9."""
        u, v = rng.standard_normal(4), rng.standard_normal(4)
        A = scale * np.outer(u, v)
        assert not mat(A).is_invertible()

    @pytest.mark.parametrize("scale", [1e-100, 1e-20, 1e-13, 1e-8, 1e8, 1e20])
    def test_inverse_scaling_equivariance(self, scale):
        """inv(cA) == inv(A)/c. BUG: inverse_optimised rejects cA as 'singular' when pivot < 1e-12."""
        A = M([[2.0, 1.0], [1.0, 3.0]])
        base = arr(A.inverse_optimised())
        assert np.allclose(arr((A * scale).inverse_optimised()) * scale, base, rtol=1e-10)

    def test_orthogonality_check_tolerance_vs_matrix_size(self):
        Q, _ = np.linalg.qr(rng.standard_normal((40, 40)))
        assert mat(Q).is_orthogonal()
        Q[0, 0] += 1e-6                                   # a genuinely perturbed 'orthogonal' matrix
        assert not mat(Q).is_orthogonal()


# ================================================================ Stress / stability
@pytest.mark.stress
class TestIllConditionedStress:
    @pytest.mark.parametrize("n", [2, 4, 6, 8])
    def test_hilbert_inverse_vs_exact(self, n):
        H = hilbert(n)
        ex = exact_inverse(H)                              # exact inverse of the *float* matrix
        kappa = np.linalg.cond(np.array(H))
        inv = M(H).inverse_optimised()
        err = max(abs(inv[i][j] - float(ex[i][j])) / abs(float(ex[i][j])) for i in range(n) for j in range(n))
        assert err <= 50 * n * EPS * kappa, f"max rel err {err:.2e} (cond {kappa:.1e})"

    @pytest.mark.parametrize("n", [2, 4, 6, 8, 10])
    def test_hilbert_rank_is_full(self, n):
        """Numerically full rank per LAPACK for n<=10; pivots < 1e-9 get declared zero here."""
        assert np.linalg.matrix_rank(np.array(hilbert(n))) == n
        assert FundamentalSubspaces(M(hilbert(n))).rank() == n

    @pytest.mark.parametrize("n", [4, 6, 8, 10])
    def test_hilbert_columns_gram_schmidt_loses_orthogonality_silently(self, n):
        """MGS loses orthogonality ~ eps*cond. A 'basis' is returned with no warning. Householder QR does not."""
        H = np.array(hilbert(n))
        Q = GramSchmidt.orthonormal_basis([vec(H[:, j]) for j in range(n)])
        Qn = np.array([varr(q) for q in Q])
        loss = np.linalg.norm(Qn @ Qn.T - np.eye(len(Q)))
        qr_loss = np.linalg.norm(np.linalg.qr(H)[0].T @ np.linalg.qr(H)[0] - np.eye(n))
        assert qr_loss < 1e-14                              # the reference achieves machine precision
        assert loss < 1e-8, f"orthogonality loss {loss:.2e}"

    def test_nearly_dependent_vectors_not_silently_mis_orthogonalised(self):
        vs = [V(1, 1, 1), V(1, 1, 1 + 1e-8)]
        Q = GramSchmidt.orthonormal_basis(vs)
        assert len(Q) == 1 or abs(Q[0].dot_product(Q[1])) < 1e-9

    def test_lauchli_matrix_gram_schmidt(self):
        d = 1e-8
        vs = [V(1, d, 0, 0), V(1, 0, d, 0), V(1, 0, 0, d)]
        Q = np.array([varr(q) for q in GramSchmidt.orthonormal_basis(vs)])
        assert np.linalg.norm(Q @ Q.T - np.eye(len(Q))) < 1e-8

    def test_adjugate_inverse_residual_on_moderately_ill_conditioned(self):
        A = np.array([[1, 1, 1], [1, 1 + 1e-6, 1], [1, 1, 1 + 1e-6]])
        kappa = np.linalg.cond(A)
        inv = arr(M(A.tolist()).inverse())
        assert np.linalg.norm(A @ inv - np.eye(3)) <= 100 * 3 * EPS * kappa

    def test_near_singular_but_valid_matrix_is_not_called_singular(self):
        """cond ~ 4e14 is representable; LAPACK inverts it. `inverse_optimised` raises 'singular'."""
        A = [[1.0, 1.0], [1.0, 1.0 + 1e-13]]
        assert np.isfinite(np.linalg.cond(np.array(A))) and np.linalg.cond(np.array(A)) < 1e15
        inv = M(A).inverse_optimised()
        assert np.linalg.norm(np.array(A) @ arr(inv) - np.eye(2)) < 1e-1

    @pytest.mark.parametrize("big", [1e150, 1e200])
    def test_matrix_with_huge_entries_does_not_overflow_silently(self, big):
        A = M([[big, 0.0], [0.0, big]])
        d = A.determinant_optimised()
        assert math.isinf(d) or d == big * big           # float overflow must surface as inf, never garbage
        inv = A.inverse_optimised()
        assert inv[0][0] == pytest.approx(1 / big, rel=1e-12)

    def test_unit_roundoff_boundary_classification(self):
        """Matrices at 1 +/- n*eps of orthogonality must be classified consistently."""
        c = M([[1.0, 0.0], [0.0, 1.0 + 4 * EPS]])
        assert c.is_orthogonal()                          # within any sane tolerance
        d = M([[1.0, 0.0], [0.0, 1.0 + 1e-6]])
        assert not d.is_orthogonal()

    @pytest.mark.parametrize("n", [60])
    def test_large_random_system_against_numpy(self, n):
        A = random_well_conditioned(n, 1)
        inv = M(A.tolist()).inverse_optimised()
        assert np.allclose(arr(inv), np.linalg.inv(A), rtol=1e-9, atol=1e-11)
        sign, logdet = np.linalg.slogdet(A)
        d = M(A.tolist()).determinant_optimised()
        assert math.copysign(1, d) == sign and math.log(abs(d)) == pytest.approx(logdet, rel=1e-10)

    def test_gram_schmidt_large_random_set_matches_qr_quality(self):
        k = 60
        X = rng.standard_normal((k, k))
        Q = np.array([varr(q) for q in GramSchmidt.orthonormal_basis([vec(r) for r in X])])
        assert Q.shape[0] == k and np.linalg.norm(Q @ Q.T - np.eye(k)) < 1e-9
