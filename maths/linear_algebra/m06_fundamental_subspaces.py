""" 
m06_fundamental_subspaces.py 

Implementation of the Four Fundamental Subspaces of a matrix from scratch. 

Topics covered 
-------------- 
- Reduced Row Echelon Form (RREF) 
- Rank 
- Nullity 
- Column Space 
- Row Space 
- Null Space 
- Left Null Space 
- Rank Nullity Theorem verification 
"""

from m01_vector_ops import Vector
from m03_linear_transformation import LinearTransformation

class FundamentalSubspaces:
    # Initializes the Fundamental Subspaces object for a matrix.
    # Computes and stores the Reduced Row Echelon Form (RREF)
    # together with the pivot column indices.
    def __init__(self, matrix: LinearTransformation) -> None:
        self.matrix = matrix
        self._rref, self._pivot_cols = self._compute_rref_with_pivots()

    # Computes the Reduced Row Echelon Form (RREF) 
    # using Gaussian elimination. 
    # Returns: tuple[list[list[float]], list[int]]
    # - The RREF of the matrix. 
    # - The pivot column indices.
    def _compute_rref_with_pivots(self) -> tuple[list[list[float]], list[int]]:
        # Work on a copy so the original matrix remains unchanged.
        mat = [row[:] for row in self.matrix]  
        m, n = self.matrix.shape
        # Current row where the next pivot will be placed.
        pivot_row = 0       
        pivot_cols = []
        for col in range(n):
            # Search for a non zero pivot in the current column.
            found = None
            for row in range(pivot_row, m):
                if abs(mat[row][col]) > 1e-9:
                    found = row
                    break
            # This column contains no pivot. Move to the next column.    
            if found is None:
                continue      

            # Move the pivot row into position.
            mat[pivot_row], mat[found] = mat[found], mat[pivot_row]
            
            # Normalize the pivot row so the pivot becomes 1.
            pivot_val = mat[pivot_row][col]
            mat[pivot_row] = [x / pivot_val for x in mat[pivot_row]]

            # Eliminate all other entries in the pivot column.
            for row in range(m):
                if row == pivot_row:
                    continue
                factor = mat[row][col]
                mat[row] = [mat[row][k] - factor * mat[pivot_row][k] for k in range(n)]
            pivot_cols.append(col)
            pivot_row += 1    # Move to the next pivot row.
            # Every row already has a pivot.
            if pivot_row == m:
                break          
        return mat, pivot_cols

    # Returns the rank of the matrix.
    # rank(A) = number of pivot columns.
    def rank(self) -> int:
        return len(self._pivot_cols)
    
    # Returns the nullity of the matrix. 
    # nullity(A) = number of columns − rank(A)
    def nullity(self) -> int:
        return self.matrix.shape[1] - self.rank()     
       
    # Checks whether every entry in a row is approximately zero. 
    # Used when constructing the Row Space basis.
    @staticmethod
    def _is_zero_row(row: list[float], tol: float = 1e-9) -> bool:
        return all(abs(x) < tol for x in row)

    # Extracts a column of the matrix as a Vector.
    def _extract_column(self, col_index: int) -> Vector:
        m, n = self.matrix.shape
        if col_index < 0 or col_index >= n:
            raise IndexError(f"Column index {col_index} out of range for shape {self.matrix.shape}.")
        return Vector([self.matrix[row][col_index] for row in range(m)])
    
    # Extracts a row of the matrix as a Vector.
    def _extract_row(self, row_index: int) -> Vector:
        m, _ = self.matrix.shape                       
        if row_index < 0 or row_index >= m:
            raise IndexError(f"Row index {row_index} out of range for shape {self.matrix.shape}.")
        return Vector(self.matrix[row_index])
    
    # Computes a basis for the Column Space. 
    # The basis consists of the pivot columns from the original matrix.
    def column_space(self) -> list[Vector]:
        return [self._extract_column(col) for col in self._pivot_cols]
    
    # Computes a basis for the Row Space. 
    # The basis consists of the non-zero rows of the Reduced Row Echelon Form.
    def row_space(self) -> list[Vector]:
        return [Vector(row) for row in self._rref              
                if not FundamentalSubspaces._is_zero_row(row)]
    
    # Computes a basis for the Null Space. 
    # The Null Space consists of all solutions to the homogeneous system, Ax = 0
    def null_space(self) -> list[Vector]:
        m, n = self.matrix.shape                                         
        free_cols = [col for col in range(n) if col not in self._pivot_cols]
        if not free_cols:
            return []     # only trivial solution x = 0

        basis = []
        for free_col in free_cols:
            # Construct one basis vector for each free variable.
            # Assign the current free variable the value 1 & all remaining free variables the value 0.
            basis_vector = [0.0] * n
            basis_vector[free_col] = 1.0
            # Compute every pivot variable using the Reduced Row Echelon Form.
            # and back substitute: x_pivot = -rref[pivot_row][free_col]
            for pivot_row, pivot_col in enumerate(self._pivot_cols):
                basis_vector[pivot_col] = -self._rref[pivot_row][free_col]
            basis.append(Vector(basis_vector))
        return basis

    # Computes a basis for the Left Null Space. 
    # The Left Null Space is the Null Space # of the transpose: Aᵀx = 0
    def left_null_space(self) -> list[Vector]:
        return FundamentalSubspaces(self.matrix.transpose()).null_space()
    
if __name__ == "__main__":
    # Reads a matrix from standard input.
    def input_matrix(name: str) -> LinearTransformation:
        rows = int(input(f"Enter number of rows for {name}: "))
        cols = int(input(f"Enter number of columns for {name}: "))
        matrix = []
        print(f"Enter the elements of {name} row by row:")
        for i in range(rows):
            row = list(map(float, input(f"  Row {i+1}: ").split()))
            if len(row) != cols:
                raise ValueError(f"Expected {cols} elements, got {len(row)}.")
            matrix.append(row)
        return LinearTransformation(matrix)

    A = input_matrix("A")
    fs = FundamentalSubspaces(A)

    print("\nMatrix A:")
    print(A)

    print(f"\nRank: {fs.rank()}")
    print(f"Nullity: {fs.nullity()}")
    print(f"Shape: {A.shape}")

    # Verify the Rank Nullity Theorem.
    m, n = A.shape
    print(f"\nRank Nullity Check (rank + nullity == n cols): {fs.rank()} + {fs.nullity()} = {n} → {fs.rank() + fs.nullity() == n}")
    print(f"Left Rank Nullity  (rank + dim(LNS) == m rows): {fs.rank()} + {len(fs.left_null_space())} = {m} → {fs.rank() + len(fs.left_null_space()) == m}")

    print("\nColumn Space basis (original columns at pivot positions):")
    cs = fs.column_space()
    if cs:
        for i, v in enumerate(cs):
            print(f"c{i+1} = {v}")
    else:
        print("None (zero matrix)")

    print("\nRow Space basis (non zero rows of RREF):")
    rs = fs.row_space()
    if rs:
        for i, v in enumerate(rs):
            print(f"r{i+1} = {v}")
    else:
        print("None (zero matrix)")

    print("\nNull Space basis (solutions to Ax = 0):")
    ns = fs.null_space()
    if ns:
        for i, v in enumerate(ns):
            print(f"n{i+1} = {v}")
    else:
        print("Trivial only — x = 0 (full rank, no free variables)")

    print("\nLeft Null Space basis (solutions to A^T x = 0):")
    lns = fs.left_null_space()
    if lns:
        for i, v in enumerate(lns):
            print(f"l{i+1} = {v}")
    else:
        print("Trivial only — x = 0 (full row rank)")

    # Verify that every Null Space basis vector satisfies Ax = 0.
    print("\nNull Space sanity check (A @ n_i should be zero vector):")
    for i, v in enumerate(fs.null_space()):
        result = A.apply_transformation(v)
        is_zero = all(abs(x) < 1e-9 for x in result)
        print(f"  A @ n{i+1} = {result} → {'✓' if is_zero else '✗'}")

    # Verify that every Left Null Space basis vector satisfies Aᵀx = 0.
    print("\nLeft Null Space sanity check (A^T @ l_i should be zero vector):")
    At = A.transpose()
    for i, v in enumerate(fs.left_null_space()):
        result = At.apply_transformation(v)
        is_zero = all(abs(x) < 1e-9 for x in result)
        print(f"  A^T @ l{i+1} = {result} → {'✓' if is_zero else '✗'}")


        








        

