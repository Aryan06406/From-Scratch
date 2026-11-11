"""
Asymptotic / scaling tests: does the measured growth match the documented complexity,
and do the public convenience methods stay usable at realistic sizes?

Exponents are fitted on log(time) vs log(n).
Timings use warm up, batched execution and median measurements.
Thresholds allow for machine dependent timing variation.
"""
import numpy as np
import pytest

from maths.linear_algebra.core import (FundamentalSubspaces, GramSchmidt, LinearTransformation,
                                       Projection, Vector)
from maths.test_helpers import (M, best_time, growth_exponent, mat, time_limit, vec)


def rand_mat(n, m=None, seed=0):
    return mat(np.random.default_rng(seed).standard_normal((n, m or n)))


def rand_vec(n, seed=0):
    return vec(np.random.default_rng(seed).standard_normal(n))


def exponent(fn_of_n, sizes, repeat=9):
    times = []
    for n in sizes:
        fn = fn_of_n(n)
        measured_time = best_time(fn, repeat)
        times.append(measured_time)

    result = growth_exponent(sizes, times)

    print("\nSCALING TEST")
    for n, t in zip(sizes, times):
        print(f"n={n:>8} | time={t:.8f}s")
    print(f"Measured exponent: {result:.4f}")

    return result


@pytest.mark.performance
class TestPolynomialAlgorithms:
    @pytest.mark.parametrize("name", ["dot_product", "hadamard_product", "__add__", "distance_between_vectors"])
    def test_vector_binary_ops_are_linear(self, name):
        def make(n):
            a, b = rand_vec(n, 1), rand_vec(n, 2)
            return lambda: getattr(a, name)(b)
        assert 0.5 <= exponent(make, [20_000, 40_000, 80_000, 160_000]) <= 1.35

    @pytest.mark.parametrize("name", ["magnitude", "unit_vector", "angle_between_vectors"])
    def test_vector_unary_ops_are_linear(self, name):
        def make(n):
            a = rand_vec(n, 1)
            if name == "angle_between_vectors":
                return lambda: a.angle_between_vectors(a)
            return lambda: getattr(a, name)()      
        measured = exponent(make, [20_000, 40_000, 80_000, 160_000])
        upper_bound = 1.6 if name == "magnitude" else 1.35
        assert 0.8 <= measured <= upper_bound

    def test_matrix_multiplication_is_cubic(self):
        def make(n):
            A, B = rand_mat(n, seed=1), rand_mat(n, seed=2)
            return lambda: A * B
        assert 2.5 <= exponent(make, [20, 40, 80], 2) <= 3.5

    @pytest.mark.parametrize("op", ["apply", "transpose", "add"])
    def test_matrix_quadratic_operations(self, op):
        def make(n):
            A, B, x = rand_mat(n, seed=1), rand_mat(n, seed=2), rand_vec(n)
            return {"apply": lambda: A.apply_transformation(x),
                    "transpose": A.transpose,
                    "add": lambda: A + B}[op]
        upper = 2.8 if op == "transpose" else 2.5     
        assert 1.6 <= exponent(make, [100, 200, 400, 800]) <= upper

    def test_determinant_optimised_is_cubic(self):
        def make(n):
            A = rand_mat(n, seed=3)
            return A.determinant_optimised
        assert 2.5 <= exponent(make, [40, 80, 160], 7) <= 3.6

    def test_inverse_optimised_is_cubic(self):
        def make(n):
            A = rand_mat(n, seed=4)
            return A.inverse_optimised
        assert 2.5 <= exponent(make, [20, 40, 80], 2) <= 3.7

    def test_rank_computation_is_cubic(self):
        def make(n):
            A = rand_mat(n, seed=5)
            return lambda: FundamentalSubspaces(A)
        assert 2.5 <= exponent(make, [30, 60, 120], 2) <= 3.6

    def test_gram_schmidt_is_quadratic_in_count_linear_in_dimension(self):
        def by_count(k):
            vs = [rand_vec(100, s) for s in range(k)]
            return lambda: GramSchmidt.orthonormal_basis(vs)
        def by_dim(d):
            vs = [rand_vec(d, s) for s in range(30)]
            return lambda: GramSchmidt.orthonormal_basis(vs)
        assert 1.6 <= exponent(by_count, [20, 40, 80], 2) <= 2.6
        assert 0.6 <= exponent(by_dim, [200, 400, 800], 2) <= 1.4

    def test_projection_matrix_is_quadratic(self):
        def make(n):
            v = rand_vec(n)
            return lambda: Projection.projection_matrix(v)
        assert 1.6 <= exponent(make, [100, 200, 400, 800]) <= 2.5


@pytest.mark.scaling
class TestFactorialPathsAreUsable:
    """`determinant`, `inverse`, `is_invertible` use O(n!) cofactor expansion."""

    def test_cofactor_recursion_is_factorial_documented_characterisation(self):
        calls = {"n": 0}
        orig = LinearTransformation._det          # the recursive worker (determinant() just delegates to it)

        def counting(matrix):
            calls["n"] += 1
            return orig(matrix)
        LinearTransformation._det = staticmethod(counting)
        try:
            counts = []
            for n in range(3, 8):
                calls["n"] = 0
                rand_mat(n, seed=n).determinant()
                counts.append(calls["n"])
        finally:
            LinearTransformation._det = staticmethod(orig)
        ratios = [b / a for a, b in zip(counts, counts[1:])]
        assert all(r > n for r, n in zip(ratios, range(3, 8))), f"call counts {counts}"   # ~ n!  growth

    @pytest.mark.parametrize("n", [9, 10, 12])
    def test_is_invertible_finishes_quickly(self, n):
        """BUG: is_invertible() calls the O(n!) determinant(); 12x12 would take minutes."""
        A = LinearTransformation.identity(n) * 2.0
        with time_limit(3):
            assert A.is_invertible()

    @pytest.mark.parametrize("n", [8, 10, 11])
    def test_inverse_finishes_quickly(self, n):
        """BUG: inverse() = n^2 cofactors x O(n!) determinants."""
        A = mat(np.random.default_rng(n).standard_normal((n, n)) + n * np.eye(n))
        with time_limit(3):
            A.inverse()

    def test_optimised_paths_handle_sizes_the_cofactor_paths_cannot(self):
        n = 100
        A = rand_mat(n, seed=9)
        with time_limit(20):
            d = A.determinant_optimised()
            inv = A.inverse_optimised()
        assert np.isfinite(d) and np.allclose(np.array(inv.matrix) @ np.array(A.matrix), np.eye(n), atol=1e-8)


def test_vector_operation_diagnostic():
    import timeit

    a = rand_vec(160_000, 1)
    b = rand_vec(160_000, 2)

    operations = {
        "dot_product": lambda: a.dot_product(b),
        "magnitude": lambda: a.magnitude(),
        "unit_vector": lambda: a.unit_vector(),
    }

    for name, operation in operations.items():
        print(f"\n{name}")

        # Warm up
        result = operation()

        print("Result type:", type(result))
        print("Result:", str(result)[:100])

        timer = timeit.Timer(operation)
        print("100 calls:", timer.timeit(number=100))
        print("1000 calls:", timer.timeit(number=1000))        