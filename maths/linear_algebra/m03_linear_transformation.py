import math
from m01_vector_ops import Vector

class LinearTransformation:
    # Initialises from a list of lists (rows) & all rows must have equal length
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            raise ValueError("Matrix cannot be empty.")
        col = len(matrix[0])
        for row in matrix:
            if len(row) != col:
                raise ValueError("All rows must have same no. of columns.")
        self.matrix = [list(row) for row in matrix]
    
    # Representation
    # Prints each row on a new line, readable output
    def __str__(self):
        rows = [str(row) for row in self.matrix]
        return "\n".join(rows)
    
    # Gives the unambiguous version for debugging
    def __repr__(self):
        return f"LinearTransformation({self.matrix})"
    
    # Shape & access
    # To check dimensions before operations
    @property
    def shape(self):
        return len(self.matrix), len(self.matrix[0])
    
    # Lets you do A[0] to get the first row
    def __getitem__(self, index):
        return self.matrix[index]
    
    # Lets you do A == B
    def __eq__(self, other):
        if not isinstance(other, LinearTransformation):
            return NotImplemented
        return self.matrix == other.matrix
    
    # Dimension check
    # Called before addition/subtraction to check if they are of same shape or not    
    def _check_same_shape(self, other):
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        
    # Called before trace, determinant & inverse as it checks whther the matrix is
    # square or not     
    def _check_square(self):
        r, c = self.shape
        if r != c:
            raise ValueError(f"Operation requires a square matrix, got {self.shape}.")
        
    # Arithmetic
    # Addition
    def __add__(self, other):
        self._check_same_shape(other)
        result = []
        for r1, r2 in zip(self.matrix, other.matrix):
            new_row = []
            for a, b in zip(r1, r2):
                new_row.append(a + b)
            result.append(new_row)    
        return LinearTransformation(result)
    
    # Subtraction
    def __sub__(self, other):
        self._check_same_shape(other)
        result = []
        for r1, r2 in zip(self.matrix, other.matrix):
            new_row = []
            for a, b in zip(r1, r2):
                new_row.append(a - b)
            result.append(new_row)    
        return LinearTransformation(result)
    
    # Multiplication
    def __mul__(self, other):
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
    
    # Just calls __mul__ so that 3 * A and A * 3 works both same
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    # Apply transformation to a vector, return M * v as a new vector
    # No of columns in M must be equal to the dimension of v
    def apply(self, vector):
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
    
    # Transpose
    def transpose(self):
        r, c = self.shape
        result = []
        for j in range(c):
            new_row = []
            for i in range(r):
                new_row.append(self.matrix[i][j])
            result.append(new_row)
        return LinearTransformation(result)

    # Trace (Sum of diagonal elements, requires a square matrix)
    def trace(self):
        self._check_square()
        return sum(self.matrix[i][i] for i in range(self.shape[0]))
    
    # Determinant 
    # Brute force way (recursive cofactor expansion) - Time complexity: O(n!)
    def determinant(self, matrix = None):
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
    
    # Optimised way (Gaussian elimination) - Time complexity: O(n^3)
    def determinant_optimised(self, matrix = None):
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
    # Brute Force (Adjoint method): A^(-1) = (1 / det(A)) * adj(A) - Time complexity: O(n!)
    def inverse(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is invertible.")
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

    # Optimised method (Gauss Jordan elimination) - Time complexity: O(n^3)
    def inverse_optimised(self):
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
    # Call cls(...) instead of LinearTransformation(...) so subclasses inherit them correctly.
    @classmethod
    def identity(cls, n):      
        return cls([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    
    @classmethod
    def zeros(cls, rows, cols):
        return cls([[0] * cols for _ in range(rows)])
    
    # Structural checks
    # A square matrix is invertible if its determinant is non zero.
    def is_invertible(self):
        try:
            self._check_square()
        except ValueError:
            return False
        return abs(self.determinant()) > 1e-9
 
    # M is symmetric if M == M^T. Requires a square matrix.
    def is_symmetric(self):
        try:
            self._check_square()
        except ValueError:
            return False
        return self == self.transpose()
    
    # M is orthogonal if M @ M^T == I (equivalently M^T == M^{-1}).
    # This holds when every column is a unit vector and columns are mutually orthogonal.
    def is_orthogonal(self):
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
    A = LinearTransformation([[1, 2], [3, 4]])
    B = LinearTransformation([[5, 6], [7, 8]])
    v = Vector([1, 0])
 
    print("A =\n", A)
    print("\nShape:", A.shape)
    print("Trace:", A.trace())
    print("Determinant:", A.determinant_optimised())
    print("Is invertible:", A.is_invertible())
    print("Is symmetric:", A.is_symmetric())
 
    print("\nA + B =\n", A + B)
    print("\nA * B =\n", A * B)
    print("\nA transposed =\n", A.transpose())
    print("\nA inverse =\n", A.inverse_optimised())
    print("\nApply A to v:", A.apply(v))
 
    print("\nIdentity (3x3):\n", LinearTransformation.identity(3))
    print("\nZeros (2x3):\n", LinearTransformation.zeros(2, 3))    
    


    
