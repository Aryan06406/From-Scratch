# From Scratch: Maths & Machine Learning

Mathematics (and, next, classical machine learning) implemented **from first principles** in pure Python, with the goal of understanding every algorithm well enough to derive it, implement it, break it, and test it against trusted references.

> **Status:** Linear Algebra and Probability are implemented and tested. Statistics and the machine-learning layer are planned (see [Roadmap](#roadmap)).

---

## Table of contents

1. [Philosophy](#philosophy)
2. [What's implemented](#whats-implemented)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Project architecture](#project-architecture)
6. [Module reference](#module-reference)
7. [Numerical design decisions](#numerical-design-decisions)
8. [Testing](#testing)
9. [Visualizations](#visualizations)
10. [Continuous integration](#continuous-integration)
11. [Known issues](#known-issues)
12. [Roadmap](#roadmap)
13. [License](#license)

---

## Philosophy

- **Pure Python core.** The library code in `core/` has zero runtime dependencies: no NumPy inside the algorithms. NumPy and SciPy-style references are used only in *tests*, as independent oracles.
- **Correctness before speed.** Every algorithm is written to be readable and numerically careful. Where a textbook algorithm is fragile (e.g. `acos` for angles, the adjugate inverse), the code uses a stable alternative and explains why.
- **Test against the truth.** Implementations are verified with example-based, property-based (Hypothesis), reference (NumPy / exact `Fraction` arithmetic), numerical-precision, and scaling tests.
- **Visualize only where geometry is the point.** Linear algebra is geometric, so a few plots genuinely help (projection, Gram-Schmidt, subspaces). Plotting is a learning aid, not a deliverable: it is kept separate from the `core/` code (and is due to move to `examples/`) and is not extended by default (see [Visualizations](#visualizations)).
- **Naive and optimised side by side.** Several operations ship in two forms (e.g. determinant by cofactor expansion `O(n!)` and by Gaussian elimination `O(n³)`) so the complexity difference can be measured, not just stated.

---

## What's implemented

### Linear algebra: `maths/linear_algebra`

| Module | Class | Topics |
|---|---|---|
| `m01_vector_ops` | `Vector` | Arithmetic, scalar ops, dot / cross / Hadamard products, magnitude, unit vector, angle, distance, orthogonality, parallelism |
| `m02_norms` | `Norms` | L1, L2, L∞, general p-norm |
| `m03_linear_transformation` | `LinearTransformation` | Matrix arithmetic, matrix-vector and matrix-matrix products, transpose, trace, determinant (cofactor and elimination), inverse (adjugate and Gauss-Jordan), structural checks (invertible, symmetric, orthogonal) |
| `m04_projection` | `Projection` | Scalar projection, vector projection, rejection, orthogonal component, projection matrix |
| `m05_gram_schmidt` | `GramSchmidt` | Orthogonalisation, orthogonal and orthonormal bases, verification report |
| `m06_fundamental_subspaces` | `FundamentalSubspaces` | RREF, rank, nullity, column / row / null / left-null spaces |

### Probability: `maths/probability`

| Module | Classes | Topics |
|---|---|---|
| `m00_probability_foundations` | `SampleSpace`, `Event`, `SetOperations`, `ProbabilityMeasure`, `ConditionalProbability`, `RandomVariable` | Finite sample spaces, events as sets, set algebra, probability measures, conditional probability, random variables as functions on outcomes |
| `m01_bayes_theorem` | `BayesTheorem` | Posterior from events; posterior from a prior/likelihood partition (law of total probability) |
| `m02_distributions` | see below | PMF/PDF, CDF, mean, variance, standard deviation |

**Distributions (`m02_distributions`)**

- Discrete: `BernoulliDistribution`, `BinomialDistribution`, `GeometricDistribution`, `PoissonDistribution`
- Continuous: `ContinuousUniformDistribution`, `ExponentialDistribution`, `GaussianDistribution`, `GammaDistribution`, `BetaDistribution`
- Supporting numerics: regularized incomplete gamma and beta functions, composite Simpson integration

### Statistics: `maths/statistics`

Placeholder package. Not implemented yet.

---

## Installation

Requires **Python 3.11 or 3.12**.

```bash
git clone <your-repo-url>
cd From-Scratch

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"            # library + pytest + hypothesis
```

`pyproject.toml` currently declares `numpy` and `matplotlib` as dependencies. In practice the library itself needs neither: NumPy is used only by tests and visualizations, and matplotlib only by the visualization scripts. The planned cleanup makes the core zero-dependency, with `matplotlib` as an optional extra (`pip install -e ".[dev,viz]"`). `requirements.txt` mirrors the current dependencies for CI.

---

## Quick start

Run everything from the **repository root**.

### Use the library

```python
from maths.linear_algebra import Vector, LinearTransformation, Projection, GramSchmidt, FundamentalSubspaces

# Vectors
a, b = Vector([1, 2, 3]), Vector([4, 5, 6])
a + b                      # [5, 7, 9]
a.dot_product(b)           # 32
a.cross_product(b)         # [-3, 6, -3]
a.angle_between_vectors(b) # degrees

# Matrices
A = LinearTransformation([[2, 1], [5, 3]])
A.determinant_optimised()  # 1.0 (O(n^3) elimination)
A.inverse_optimised()      # Gauss-Jordan inverse
A.apply_transformation(Vector([1, 1]))

# Projection
Projection.vector_projection(Vector([3, 4]), Vector([5, 1]))

# Gram-Schmidt
basis = GramSchmidt.orthonormal_basis([Vector([1, 1]), Vector([1, 0])])

# Four fundamental subspaces
fs = FundamentalSubspaces(LinearTransformation([[1, 2, 3], [2, 4, 6]]))
fs.rank(), fs.nullity()
fs.null_space(), fs.column_space(), fs.row_space(), fs.left_null_space()
```

```python
from maths.probability import SampleSpace, Event, ProbabilityMeasure, ConditionalProbability, BayesTheorem
from maths.probability.core.m02_distributions import GaussianDistribution, BinomialDistribution

# Finite probability
omega = SampleSpace({1, 2, 3, 4, 5, 6})
even = Event("Even", {2, 4, 6}, omega)
big = Event("Greater than 3", {4, 5, 6}, omega)
P = ProbabilityMeasure(omega)
P(even & big)                                   # 1/3
ConditionalProbability(P).compute(even, big)    # 2/3

# Bayes (medical-test example)
BayesTheorem.compute_from_partition(
    priors={"Disease": 0.01, "Healthy": 0.99},
    likelihoods={"Disease": 0.95, "Healthy": 0.05},
    target_hypothesis="Disease",
)                                               # ~0.161

# Distributions
GaussianDistribution(0, 1).cdf(1.96)            # ~0.975
BinomialDistribution(10, 0.5).pmf(5)            # ~0.246
```

> The distribution classes are not re-exported from `maths.probability`; import them from `maths.probability.core.m02_distributions` as shown above.

### Run a module's interactive demo

Every module has a `__main__` block that prompts for input and prints results:

```bash
python -m maths.linear_algebra.core.m01_vector_ops
python -m maths.linear_algebra.core.m06_fundamental_subspaces
python -m maths.probability.core.m01_bayes_theorem      # non-interactive worked examples
```

---

## Project architecture

```
From-Scratch/
├── maths/
│   ├── __init__.py
│   ├── conftest.py              # Hypothesis profiles + pytest marker registration
│   ├── test_helpers.py          # shared builders, exact references, timing, strategies
│   │
│   ├── linear_algebra/
│   │   ├── core/                # the library (pure Python)
│   │   │   ├── m01_vector_ops.py
│   │   │   ├── m02_norms.py
│   │   │   ├── m03_linear_transformation.py
│   │   │   ├── m04_projection.py
│   │   │   ├── m05_gram_schmidt.py
│   │   │   └── m06_fundamental_subspaces.py
│   │   ├── tests/               # t0x_* (examples) and a0x_* (adversarial suites)
│   │   └── visualization/       # v0x_* matplotlib companions (frozen; to move to examples/)
│   │
│   ├── probability/
│   │   ├── core/
│   │   │   ├── m00_probability_foundations.py
│   │   │   ├── m01_bayes_theorem.py
│   │   │   └── m02_distributions.py
│   │   └── tests/
│   │
│   └── statistics/              # placeholder
│
├── pyproject.toml               # packaging, dependencies, pytest config
├── requirements.txt
└── .github/workflows/           # CI
```

### Naming conventions

| Prefix | Meaning |
|---|---|
| `m0x_` | Module (core implementation), numbered in teaching order |
| `v0x_` | Visualization for the matching `m0x_` module |
| `t0x_` | Basic example-based tests for the matching `m0x_` module |
| `a0x_` | "Adversarial" test suites (boundary, properties, numerics, scaling) shared across a package |

### Dependency graph (linear algebra)

```
m01 Vector ──┬──> m02 Norms
             │
             └──> m03 LinearTransformation ──┬──> m04 Projection ──> m05 GramSchmidt
                                             │
                                             └──> m06 FundamentalSubspaces
```

`m04` depends on `m01` and `m03` (the projection matrix is returned as a `LinearTransformation`); `m05` depends on `m01` and `m04`; `m06` depends on `m01` and `m03`.

### Dependency graph (probability)

```
m00 foundations ──┬──> m01 BayesTheorem
                  └──> m02 distributions   (uses ProbabilityMeasure, SampleSpace, Event, RandomVariable)
```

### Package exports

`maths.linear_algebra` re-exports `Vector`, `Norms`, `LinearTransformation`, `Projection`, `GramSchmidt`, `FundamentalSubspaces`.
`maths.probability` re-exports `SampleSpace`, `Event`, `SetOperations`, `ProbabilityMeasure`, `ConditionalProbability`, `BayesTheorem`.

---

## Module reference

### `Vector` (`m01`)

Immutable-by-convention list of real numbers (validated on construction). Supports `+`, `-`, `*` (scalar), `/`, `//`, unary `-`, `==`, `len`, iteration, and indexing.

| Method | Description |
|---|---|
| `dot_product(other)` | Σ aᵢbᵢ |
| `cross_product(other)` | R³ only |
| `hadamard_product(other)` | Element-wise product |
| `magnitude()` | Euclidean norm via `math.hypot` (overflow/underflow-safe) |
| `unit_vector()` / `normalize()` | v / ‖v‖ |
| `angle_between_vectors(other)` | Degrees, via Kahan's `atan2` formula |
| `distance_between_vectors(other)` | ‖a − b‖ |
| `is_orthogonal(other, tol)` | Scale-invariant (compares cos θ) |
| `is_parallel(other, tol)` | Unit vectors coincide or are opposite |

### `Norms` (`m02`)

`l1_norm()`, `l2_norm()`, `infinity_norm()`, `p_norm(p)`. The p-norm rescales by the largest entry to avoid overflow.

### `LinearTransformation` (`m03`)

A matrix stored as a list of rows.

| Area | Methods |
|---|---|
| Construction | `LinearTransformation(rows)`, `identity(n)`, `zeros(r, c)` |
| Arithmetic | `+`, `-`, `*` (scalar or matrix), `transpose()`, `trace()`, `shape` |
| Application | `apply_transformation(vector)` computes Mv |
| Determinant | `determinant()` is `O(n!)` cofactor expansion; `determinant_optimised()` is `O(n³)` elimination with partial pivoting |
| Inverse | `inverse()` uses the adjugate for n ≤ 6 and falls back to Gauss-Jordan beyond that; `inverse_optimised()` is Gauss-Jordan with partial pivoting, `O(n³)` |
| Checks | `is_invertible()`, `is_symmetric()`, `is_orthogonal()` |

### `Projection` (`m04`)

Static methods: `scalar_projection(u, v)`, `vector_projection(u, v)`, `rejection(u, v)`, `orthogonal_component(u, v)`, `projection_matrix(v)`.

### `GramSchmidt` (`m05`)

Static methods: `orthogonalize_vector(v, basis)`, `orthogonal_basis(vectors)`, `orthonormal_basis(vectors)`, `is_orthogonal_basis(vectors)`, `is_orthonormal_basis(vectors)`, `verify(original, basis)`. Linearly dependent inputs are dropped automatically.

### `FundamentalSubspaces` (`m06`)

Constructed from a `LinearTransformation`; computes RREF and pivot columns once. Provides `rank()`, `nullity()`, `column_space()`, `row_space()`, `null_space()`, `left_null_space()`. Rank-nullity is checked in the interactive demo.

### Probability (`m00`–`m02`)

- `SampleSpace(outcomes)` is a finite set of hashable outcomes.
- `Event(name, outcomes, sample_space)` is a subset of the space; supports `|` (union), `&` (intersection), `-` (difference), `~` (complement).
- `ProbabilityMeasure(sample_space, weights=None)`: uniform if no weights; callable as `P(event)`.
- `ConditionalProbability(P).compute(A, B)`: P(A | B), raising if P(B) = 0.
- `RandomVariable(sample_space, mapping)`: `event_equals(v)`, `event_less_than_or_equal(v)`.
- `BayesTheorem`: `compute_from_events(H, E)` and the static `compute_from_partition(priors, likelihoods, target)`.
- All distributions expose `pmf`/`pdf`, `cdf`, `expected_value`, `variance`, `std_dev`.

---

## Numerical design decisions

These are the places where the code deliberately departs from the textbook formula.

| Problem | Naive approach | What this repo does |
|---|---|---|
| Angle between vectors | `acos(a·b / ‖a‖‖b‖)` loses precision near 0° and 180° | Kahan's `2·atan2(‖â−b̂‖, ‖â+b̂‖)` |
| Vector length | `sqrt(Σx²)` overflows or underflows | `math.hypot` |
| p-norm | `(Σ|x|ᵖ)^(1/p)` overflows | Scale by max entry first |
| Projection | `(u·v)/(v·v) · v` can overflow | Rescale v to entries in [−1, 1] first |
| Gram-Schmidt | One pass loses orthogonality ~ ε·cond(A) | Modified Gram-Schmidt, applied **twice** ("twice is enough") |
| Linear-dependence test | Fixed absolute cutoff | Relative to input norm |
| Singularity / rank | `abs(det) > 1e-9` is scale-dependent | Pivot tolerance `n · ε · ‖A‖_F` (same idea as `numpy.linalg.matrix_rank`) |
| Pivoting | Take first non-zero pivot | Partial pivoting (largest entry) |
| Adjugate inverse | Cancellation when det is tiny | Newton-Schulz refinement `X ← X + X(I − AX)` when the residual is large |
| Binomial / Poisson PMF | Factorials overflow | Computed in log space with `lgamma` |
| Binomial / Poisson CDF | Sum from 0 underflows | Sum the shorter tail from its largest term, using `math.fsum` |
| Gamma / Beta CDF | Numerical integration is slow and inaccurate | Series + continued fraction (modified Lentz) for the regularized incomplete gamma and beta functions |
| Normal CDF | `1 − tail` cancels | `0.5 · erfc(−z/√2)` |
| Probability sums | Plain `sum` accumulates error | `math.fsum` |

### Tolerances

Tolerances are scale-aware wherever possible (relative to a norm) rather than absolute. A few checks still use fixed absolute tolerances (`is_orthogonal` on matrices, the `1e-9` checks in `GramSchmidt` verification); these are documented in the code and are a known inconsistency (see [Known issues](#known-issues)).

---

## Testing

The suite has **608 tests**: 572 regular tests (about 80 s on a laptop) and 36 timing-based scaling / performance tests (about 55 s).

### Running tests

```bash
# Everything except timing-sensitive tests (recommended default)
python -m pytest -m "not performance and not scaling" -v

# Timing / complexity tests only
python -m pytest -m "performance or scaling" -v

# One package or one file
python -m pytest maths/linear_algebra/tests -v
python -m pytest maths/probability/tests/a03_numerics.py -v

# Deeper property-based search (1500 examples per property instead of 150)
HYPOTHESIS_PROFILE=thorough python -m pytest -m "not performance and not scaling"          # macOS / Linux
$env:HYPOTHESIS_PROFILE="thorough"; python -m pytest -m "not performance and not scaling"  # PowerShell
```

### How tests are organised

| File | Purpose |
|---|---|
| `t0x_*.py` | A handful of readable example-based tests per module: the "does it work at all" layer |
| `a01_unit_boundary.py` | Edge cases: empty / zero / one-element inputs, invalid types, error messages, boundary values |
| `a02_properties.py` | **Hypothesis** property-based tests: algebraic laws (commutativity, distributivity, `(AB)ᵀ = BᵀAᵀ`, rank-nullity, probability axioms, and so on) over randomly generated inputs |
| `a03_numerics.py` | **Reference tests** against NumPy and exact `Fraction` arithmetic, ill-conditioned inputs (Hilbert matrices), extreme magnitudes, and precision limits |
| `a04_scaling.py` | **Complexity tests**: fit the growth exponent of log(time) vs log(n) and assert it matches the documented big-O (for example a dot product ~ 1, matrix multiply ~ 3) |

Shared infrastructure lives in `maths/test_helpers.py`: matrix/vector builders, `exact_det` and `exact_inverse` using `Fraction`, a cross-platform `time_limit` guard (SIGALRM on POSIX, watchdog thread on Windows), a `best_time` benchmark helper, `growth_exponent`, and reusable Hypothesis strategies (`square_mats`, `diag_dominant`, `vec_pairs`, ...).

### Pytest markers

`unit`, `boundary`, `property`, `invariance`, `numeric`, `reference`, `scaling`, `stress`, `performance`. They are registered in `conftest.py`. Select with `-m`, for example `pytest -m property`.

### Hypothesis profiles

Defined in `maths/conftest.py`:

- `adv` (default): 150 examples per property, no deadline
- `thorough`: 1500 examples per property

Select with the `HYPOTHESIS_PROFILE` environment variable.

### Coverage

```bash
pip install pytest-cov
python -m pytest -m "not performance and not scaling" --cov=maths.linear_algebra.core --cov=maths.probability.core --cov-report=term-missing
```

Overall core coverage is roughly 78%. Most uncovered lines are the interactive `if __name__ == "__main__":` demo blocks rather than library logic.

---

## Visualizations

### Policy

Visualization is a **learning aid, not a goal of this repo.** The rule:

- **Keep** the six existing scripts. They cover the topics where geometry is the whole idea.
- **Freeze** them: fix breakage, but do not extend the folder by default.
- **Add a new plot only when the concept is geometric** and a picture teaches something the formula does not. Planned: eigenvectors (what a matrix does to directions), SVD (unit circle to ellipse), PCA (principal axes on a point cloud).
- **No plots for high-dimensional algorithms** (LU, QR, Cholesky, gradient descent in many dimensions, trees, boosting). Use a short result plot in a notebook instead: loss curve, decision boundary, predictions vs. actuals.
- **Time-box:** if a visual takes more than about an hour, skip it. Derivations and reference tests matter more.

### Existing scripts

`maths/linear_algebra/visualization/` holds one matplotlib companion per module. Each script prompts for inputs (press Enter to accept defaults), draws labelled panels, and saves a PNG to `maths/linear_algebra/visualization/figures/` (git-ignored).

| Script | Illustrates | Status |
|---|---|---|
| `v01_vector_ops.py` | Vector arithmetic, dot / cross products, angles | Working |
| `v02_norms.py` | L1 / L2 / L∞ / p-norm unit balls | Working |
| `v03_linear_transformation.py` | How a matrix transforms the plane | Working |
| `v04_projection.py` | Projection, rejection, orthogonal component | **Broken import** (see [Known issues](#known-issues)) |
| `v05_gram_schmidt.py` | Step-by-step orthogonalisation | **Broken import** |
| `v06_fundamental_subspaces.py` | The four fundamental subspaces | **Broken import** |

```bash
python -m maths.linear_algebra.visualization.v01_vector_ops
```

### Planned housekeeping

1. Fix the three imports (use `from maths.linear_algebra.core import ...`).
2. Add a smoke test that imports every visualization module with `MPLBACKEND=Agg`, so a refactor cannot silently break them again.
3. Move the folder from the library package to `examples/` so the core stays free of plotting code.
4. Make `matplotlib` an optional `viz` extra.

---

## Continuous integration

`.github/workflows/python-app.yml` runs on pushes and pull requests to `master`:

1. Set up Python 3.11
2. Install `requirements.txt`, `flake8`, `pytest`
3. flake8: hard failure on syntax errors / undefined names; other style findings are reported as warnings only
4. `pytest`

---

## Known issues

Being upfront about the current state of the repo:

- **Visualizations `v04`, `v05`, `v06` fail on import.** They import `Projection`, `GramSchmidt`, and `FundamentalSubspaces` from `m01`/`m03`, where these classes no longer live. Fix: import from `maths.linear_algebra.core`. `v01`–`v03` import correctly. No test imports the visualization folder, so CI did not catch this (a smoke test is planned).
- **Elimination-based rank is not rank-revealing.** For a matrix with *nearly* dependent columns (e.g. rank 5 plus 1e-9 noise), `FundamentalSubspaces.rank()` reports full rank, while an SVD-based rank with a sensible tolerance reports 5. An SVD implementation is on the roadmap.
- **Near-singular inverses are returned silently.** `inverse()` / `inverse_optimised()` do not warn for matrices with very large condition numbers.
- **Inconsistent tolerance policy** across `determinant_optimised`, `is_symmetric`, `__eq__`, and `is_orthogonal`.
- **CI runs the timing tests too.** The default `pytest` in CI includes `scaling` / `performance` tests, which can be flaky on shared runners.
- **Performance.** The pure-Python matrix product is roughly two orders of magnitude slower than NumPy (about 270× at 120×120). This is by design for the reference layer, and is why the future ML layer will use NumPy arrays.
- **Documentation** is mostly in `#` comments rather than docstrings, so `help()` is not informative yet.

---

## Roadmap

**Linear algebra:** LU with partial pivoting (and `solve`), Cholesky, QR (Householder), eigenvalues (power iteration, QR algorithm, Jacobi), SVD, pseudoinverse, least squares, PCA, condition numbers, a `__matmul__` operator.

**Probability & statistics:** export distributions from the package, sampling (inverse-CDF, Box-Muller, rejection), quantile functions, multivariate Gaussian, entropy / KL divergence / cross-entropy, expectation and covariance, MLE, bootstrap, hypothesis tests, Monte Carlo demonstrations of the LLN and CLT.

**Visualization:** fix and freeze the existing six; add plots only for eigenvectors, SVD, and PCA; use notebook result plots (not standalone scripts) for ML algorithms. See the [policy](#visualizations).

**Classical machine learning (NumPy-array layer):** optimisation (gradient descent, momentum / Adam, Newton, conjugate gradient, L-BFGS), linear / ridge / lasso / logistic regression, KNN, Naive Bayes, K-Means, GMM (EM), decision trees, random forests, gradient boosting, SVM. Each algorithm will get a derivation, an implementation, a reference test against scikit-learn, a property test, and a benchmark on a real dataset.

---

## License

See [LICENSE.md](LICENSE.md).
