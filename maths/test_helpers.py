"""Helpers for testing maths.linear_algebra.core and related modules."""
from __future__ import annotations

import contextlib
import math
import signal
import time
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import strategies as st

from maths.linear_algebra.core import LinearTransformation, Vector

EPS = float(np.finfo(float).eps)          # 2.22e-16
U = EPS / 2                               # unit round off

# builders
def V(*xs):
    return Vector(list(xs))

def M(rows):
    return LinearTransformation(rows)

def arr(A) -> np.ndarray:
    return np.array(A.matrix, dtype=float)

def varr(v) -> np.ndarray:
    return np.array(list(v), dtype=float)

def mat(a) -> LinearTransformation:
    return LinearTransformation([[float(x) for x in row] for row in np.asarray(a)])

def vec(a) -> Vector:
    return Vector([float(x) for x in np.asarray(a)])

def hilbert(n: int) -> list[list[float]]:
    return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]

def rot2(theta: float) -> LinearTransformation:
    c, s = math.cos(theta), math.sin(theta)
    return M([[c, -s], [s, c]])

# exact (Fraction) refs
def exact_det(rows) -> Fraction:
    a = [[Fraction(x) for x in r] for r in rows]
    n, det = len(a), Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if a[r][c] != 0), None)
        if p is None:
            return Fraction(0)
        if p != c:
            a[c], a[p] = a[p], a[c]
            det = -det
        det *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return det

def exact_inverse(rows) -> list[list[Fraction]]:
    n = len(rows)
    a = [[Fraction(x) for x in r] + [Fraction(int(i == j)) for j in range(n)]
         for i, r in enumerate(rows)]
    for c in range(n):
        p = next(r for r in range(c, n) if a[r][c] != 0)
        a[c], a[p] = a[p], a[c]
        pv = a[c][c]
        a[c] = [x / pv for x in a[c]]
        for r in range(n):
            if r != c and a[r][c] != 0:
                f = a[r][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [row[n:] for row in a]

# timing
class TimedOut(AssertionError):
    pass

@contextlib.contextmanager
def time_limit(seconds: float):
    """Abort a pure python computation that runs too long (POSIX only)."""
    if not hasattr(signal, "setitimer"):
        pytest.skip("needs POSIX signals")

    def _handler(signum, frame):
        raise TimedOut(f"did not finish within {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

def best_time(fn, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best

def growth_exponent(sizes, times) -> float:
    """Least squares slope of log(time) vs log(size): ~1 linear, ~2 quadratic, ~3 cubic."""
    return float(np.polyfit(np.log(sizes), np.log(times), 1)[0])

# comparisons
def rel_err(x: float, ref: float) -> float:
    if ref == 0:
        return abs(x)
    return abs(x - ref) / abs(ref)

def max_abs(a) -> float:
    return float(np.max(np.abs(a))) if np.size(a) else 0.0

# hypothesis strategies
small_int = st.integers(-20, 20)
tiny_int = st.integers(-3, 3)
dim = st.integers(1, 6)
nonzero_int = st.integers(-20, 20).filter(lambda k: k != 0)
positive_int = st.integers(1, 20)
finite_float = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)

def int_vecs(n, elems=small_int):
    return st.lists(elems, min_size=n, max_size=n).map(Vector)

def vec_pairs(elems=small_int):
    return dim.flatmap(lambda n: st.tuples(int_vecs(n, elems), int_vecs(n, elems)))

def vec_triples(elems=small_int):
    return dim.flatmap(lambda n: st.tuples(int_vecs(n, elems), int_vecs(n, elems), int_vecs(n, elems)))

def vec3(elems=small_int):
    return int_vecs(3, elems)

def float_vecs(n, lo=-100.0, hi=100.0):
    return st.lists(st.floats(lo, hi, allow_nan=False, allow_infinity=False), min_size=n, max_size=n).map(Vector)

def int_rows(r, c, elems=small_int):
    return st.lists(st.lists(elems, min_size=c, max_size=c), min_size=r, max_size=r)

def int_mats(r, c, elems=small_int):
    return int_rows(r, c, elems).map(LinearTransformation)

def square_mats(lo=1, hi=4, elems=small_int):
    return st.integers(lo, hi).flatmap(lambda n: int_mats(n, n, elems))

def diag_dominant(lo=2, hi=4):
    """Strictly diagonally dominant integer matrices: always invertible and well conditioned."""
    def build(rows):
        n = len(rows)
        out = []
        for i, r in enumerate(rows):
            r = list(r)
            r[i] = (sum(abs(x) for x in r) + 1) * (1 if i % 2 == 0 else -1)
            out.append(r)
        return LinearTransformation(out)
    return st.integers(lo, hi).flatmap(lambda n: int_rows(n, n, st.integers(-5, 5)).map(build))

def vec_lists(min_k=1, max_k=5, elems=tiny_int):
    return st.tuples(st.integers(1, 5), st.integers(min_k, max_k)).flatmap(
        lambda t: st.lists(int_vecs(t[0], elems), min_size=t[1], max_size=t[1]))
