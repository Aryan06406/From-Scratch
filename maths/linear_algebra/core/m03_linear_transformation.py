"""
m03_linear_transformation.py

Implementation of matrices and linear transformations from scratch.

Topics covered
--------------
- Matrix arithmetic
- Matrix-vector multiplication
- Matrix-matrix multiplication
- Transpose
- Trace
- Determinant (recursive & Gaussian elimination)
- Matrix inverse (Adjoint & Gauss-Jordan)
- Identity and zero matrices
- Structural matrix properties
"""

from __future__ import annotations
from maths.linear_algebra.core.m01_vector_ops import Vector

class LinearTransformation:
    # Initializes a matrix from a nested list.
    # All rows must have the same number of columns.
    def __init__(self, matrix: list[list[float]]) -> None:
        if not matrix or not matrix[0]:
            raise ValueError("Matrix cannot be empty.")
        col = len(matrix[0])
        for row in matrix:
            if len(row) != col:
                raise ValueError("All rows must have same no. of columns.")
        self.matrix = [list(row) for row in matrix]
    
    # Returns a human-readable representation of the matrix.
    def __str__(self) -> str:
        rows = [str(row) for row in self.matrix]
        return "\n".join(rows)
    
    # Returns the official representation used for debugging.
    def __repr__(self) -> str:
        return f"LinearTransformation({self.matrix})"
    
    # Allows iteration over the matrix rows.
    def __iter__(self):
        return iter(self.matrix)
    
    # Returns the dimensions of the matrix as (rows, columns).
    @property
    def shape(self) -> tuple[int, int]:
        return len(self.matrix), len(self.matrix[0])
    
    # Enables indexing using matrix[i].
    def __getitem__(self, index: int) -> list[float]:
        return self.matrix[index]
    
    # Checks whether two matrices are equal.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LinearTransformation):
            return NotImplemented
        return self.matrix == other.matrix
    
    # Dimension check
    # Verifies that two matrices have identical dimensions.
    def check_same_shape(self, other: "LinearTransformation") -> None:
        self._check_same_shape(other)

    # Internal helper for validating equal matrix dimensions.
    def _check_same_shape(self, other: "LinearTransformation") -> None:
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        
    # Called before trace, determinant & inverse as it checks whther the matrix is
    # Verifies that the matrix is square.
    def check_square(self) -> None:
        self._check_square()

    # Internal helper for validating square matrices.
    def _check_square(self) -> None:
        r, c = self.shape
        if r != c:
            raise ValueError(f"Operation requires a square matrix, got {self.shape}.")
        
    # Arithmetic
    # Computes element-wise matrix addition.
    def __add__(self, other: "LinearTransformation") -> "LinearTransformation":
        self._check_same_shape(other)
        result = []
        for r1, r2 in zip(self.matrix, other.matrix):
            new_row = []
            for a, b in zip(r1, r2):
                new_row.append(a + b)
            result.append(new_row)    
        return LinearTransformation(result)
    
    # Computes element-wise matrix subtraction.
    def __sub__(self, other: "LinearTransformation") -> "LinearTransformation":
        self._check_same_shape(other)
        result = []
        for r1, r2 in zip(self.matrix, other.matrix):
            new_row = []
            for a, b in zip(r1, r2):
                new_row.append(a - b)
            result.append(new_row)    
        return LinearTransformation(result)
    
    # Performs scalar multiplication or matrix multiplication.
    # Scalar: αA   and    Matrix: AB
    def __mul__(self, other: int | float | "LinearTransformation") -> "LinearTransformation":
        if isinstance(other, (int, float)):
            result = []
            for row in self.matrix:
                new_row = [] 
                for val in row:
                    new_row.append(val * other)
                result.append(new_row)    
            return LinearTransformation(result)
        
        if isinstance(other, LinearTransformation):
            r1, c1 = self.shape
            r2, c2 = other.shape
            if c1 != r2:
                raise ValueError(
                    f"Cannot multiply: ({r1}x{c1}) * ({r2}x{c2})."
                    "\nInner dimensions must match"
                )
            other_cols = list(zip(*other.matrix))
            result = []
            for row in self.matrix:
                result_row = []
                for col in other_cols:
                    result_row.append(sum(a * b for a, b in zip(row, col)))
                result.append(result_row)    
            return LinearTransformation(result)
        return NotImplemented
    
    # Enables scalar multiplication from the left. Example: 3 * A
    def __rmul__(self, scalar: int | float) -> "LinearTransformation":
        return self.__mul__(scalar)
    
    # Applies the linear transformation to a vector.
    # Computes: Mv where M is the matrix and v is the input vector.
    def apply_transformation(self, vector: Vector) -> Vector:
        if not isinstance(vector, Vector):
            raise TypeError("Argument must be a vector instance.")
        r, c = self.shape
        if c != len(vector):
            raise ValueError(
                f"Cannot apply ({r}x{c}) matrix to a {len(vector)} dimensional vector."
            )
        result = []
        for i in range(r):
            row_sum = 0
            for cell, val in zip(self.matrix[i], vector):
                row_sum += cell * val
            result.append(row_sum)
        return Vector(result)         
    
    # Computes the transpose of the matrix: (Aᵀ)ij = Aji
    def transpose(self) -> "LinearTransformation":
        r, c = self.shape
        result = []
        for j in range(c):
            new_row = []
            for i in range(r):
                new_row.append(self.matrix[i][j])
            result.append(new_row)
        return LinearTransformation(result)

    # Computes the trace of a square matrix: tr(A) = Σ Aii
    def trace(self) -> float:
        self._check_square()
        return sum(self.matrix[i][i] for i in range(self.shape[0]))
    
    # Determinant 
    # Computes the determinant using recursive cofactor expansion.
    # Time Complexity: O(n!)
    def determinant(self, matrix: list[list[float]] | None = None) -> float:
        if matrix is None:
            self._check_square()
            matrix = self.matrix
        n = len(matrix)
        # Base case: 1x1 matrix
        if n == 1:
            return matrix[0][0]   
        # Base case: 2x2 matrix 
        if n == 2:
            return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        det = 0
        # Cofactor expansion along the first row
        for col in range(n):
            minor = []               # Create minor matrix
            for row in range(1, n):
                minor_row = []
                for j in range(n):
                    if j != col:
                        minor_row.append(matrix[row][j])
                minor.append(minor_row)
            # Add / Subtract cofactor 
            sign = (-1) ** col
            det += sign * matrix[0][col] * self.determinant(minor)   
        return det
    
    # Computes the determinant using Gaussian elimination. 
    # Time complexity: O(n^3)
    def determinant_optimised(self, matrix = None) -> float:
        self._check_square()
        n = self.shape[0]
 
        # Work on a copy so we don't mutate self
        mat = [row[:] for row in self.matrix]
        sign = 1
 
        for col in range(n):
            # Partial pivot: find the row with the largest absolute value in this column
            pivot_row = max(range(col, n), key=lambda r: abs(mat[r][col]))
            if mat[pivot_row][col] == 0:
                return 0  # Singular matrix
 
            if pivot_row != col:
                mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
                sign *= -1  # Row swap flips the sign of the determinant
 
            # Eliminate below pivot
            for row in range(col + 1, n):
                if mat[col][col] == 0:
                    continue
                factor = mat[row][col] / mat[col][col]
                for k in range(col, n):
                    mat[row][k] -= factor * mat[col][k]
 
        # Determinant = product of diagonal elements * accumulated sign
        det = sign
        for i in range(n):
            det *= mat[i][i]
        return det

    # Inverse
    # Computes the inverse using the adjugate method: A^(-1) = (1 / det(A)) * adj(A) 
    # Time complexity: O(n!)
    def inverse(self) -> "LinearTransformation":
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        # Find cofactor matrix
        cofactors = []
        n = len(self.matrix)
        for i in range(n):
            row = []
            for j in range(n):
                minor = [
                    [self.matrix[x][y] for y in range(n) if y != j]
                    for x in range(n) if x != i
                ]
                value = ((-1) ** (i + j)) * self.determinant(minor)
                row.append(value)
            cofactors.append(row)
        adjugate = LinearTransformation(cofactors).transpose()   # Adjugate = transpose of cofactor matrix
        # Divide by determinant
        return LinearTransformation([
            [adjugate[i][j] / det for j in range(n)]
            for i in range(n)
        ])     

    # Computes the inverse using Gauss-Jordan elimination. 
    # Time complexity: O(n^3)
    def inverse_optimised(self) -> "LinearTransformation":
        self._check_square()
        n = self.shape[0]
 
        # Build augmented matrix [M | I]
        aug = [self.matrix[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
               for i in range(n)]
 
        for col in range(n):
            # Partial pivot
            pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot_row][col]) < 1e-12:
                raise ValueError("Matrix is singular and cannot be inverted.")
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
 
            # Scale pivot row so leading element = 1
            pivot_val = aug[col][col]
            aug[col] = [x / pivot_val for x in aug[col]]
 
            # Eliminate the entire column (above and below)
            for row in range(n):
                if row == col:
                    continue
                factor = aug[row][col]
                aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(2 * n)]
 
        # Extract the right half — that's M^{-1}
        inv = [aug[i][n:] for i in range(n)]
        return LinearTransformation(inv)        

    # Factory methods 
    # Creates an n × n identity matrix.
    @classmethod
    def identity(cls, n: int) -> "LinearTransformation":      
        return cls([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    
    # Creates a matrix whose entries are all zero.
    @classmethod
    def zeros(cls, rows: int, cols: int) -> "LinearTransformation":
        return cls([[0] * cols for _ in range(rows)])
    
    # Structural checks
    # Checks whether the matrix is invertible. A matrix is invertible iff det(A) ≠ 0.    
    def is_invertible(self) -> bool:
        try:
            self._check_square()
        except ValueError:
            return False
        return abs(self.determinant()) > 1e-9
 
    # Checks whether the matrix is symmetric. A = Aᵀ
    def is_symmetric(self) -> bool:
        try:
            self._check_square()
        except ValueError:
            return False
        return self == self.transpose()
    
    # Checks whether the matrix is orthogonal.A is orthogonal if
    # AAᵀ = I Equivalently, A⁻¹ = Aᵀ Every column forms an orthonormal basis.
    def is_orthogonal(self) -> bool:
        try:
            self._check_square()
        except ValueError:
            return False
        n = self.shape[0]
        product = self * self.transpose()
        identity = LinearTransformation.identity(n)
 
        # Compare element wise with tolerance for floating point
        for i in range(n):
            for j in range(n):
                if abs(product.matrix[i][j] - identity.matrix[i][j]) > 1e-9:
                    return False
        return True
    
if __name__ == "__main__":
    # Reads a matrix from standard input.
    def input_matrix(name: str) -> LinearTransformation:
        rows = int(input(f"Enter number of rows for {name}: "))
        cols = int(input(f"Enter number of columns for {name}: "))
        matrix = []
        print(f"Enter the elements of {name} row by row:")
        for i in range(rows):
            row = list(map(float, input(f"Row {i+1}: ").split()))
            if len(row) != cols:
                raise ValueError(f"Each row must contain exactly {cols} elements.")
            matrix.append(row)
        return LinearTransformation(matrix)

    # Reads a vector from standard input.
    def input_vector() -> Vector:
        size = int(input("Enter the dimension of the vector: "))
        values = list(map(float, input("Enter the vector elements: ").split()))
        if len(values) != size:
            raise ValueError("Incorrect number of vector elements.")
        return Vector(values)

    # User inputs
    A = input_matrix("A")
    B = input_matrix("B")
    v = input_vector()

    # Outputs
    print("\nA =")
    print(A)

    print("\nShape:", A.shape)

    if A.shape[0] == A.shape[1]:
        print("Trace:", A.trace())
        print("Determinant:", A.determinant_optimised())
        print("Is invertible:", A.is_invertible())
        print("Is symmetric:", A.is_symmetric())
        if A.is_invertible():
            print("\nA inverse =")
            print(A.inverse_optimised())

    if A.shape == B.shape:
        print("\nA + B =")
        print(A + B)

    if A.shape[1] == B.shape[0]:
        print("\nA * B =")
        print(A * B)

    print("\nA transposed =")
    print(A.transpose())

    if len(v) == A.shape[1]:
        print("\nApply A to v:")
        print(A.apply_transformation(v))

    n = int(input("\nEnter size for identity matrix: "))
    print("\nIdentity Matrix:")
    print(LinearTransformation.identity(n))

    rows = int(input("\nEnter rows for zero matrix: "))
    cols = int(input("Enter columns for zero matrix: "))
    print("\nZero Matrix:")
    print(LinearTransformation.zeros(rows, cols))
    


    
