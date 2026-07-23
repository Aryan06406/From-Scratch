from m01_vector_ops import Vector
from m03_linear_transformation import LinearTransformation

class FundamentalSubspaces:
    def __init__(self, matrix: LinearTransformation):
        self.matrix = matrix
        self._rref, self._pivot_cols = self._compute_rref()

    def _compute_rref(self):
        mat = [row[:] for row in self.matrix]  # copy
        m, n = self.matrix.shape
        pivot_row = 0       # next row to place pivot
        pivot_cols = []
        for col in range(n):
            # Find a non zero entry in this column below the pivot_row
            found = None
            for row in range(pivot_row, m):
                if abs(mat[row][col]) > 1e-9:
                    found = row
                    break
            if found is None:
                continue      # Entire column below is zero, skip the column

            # Swap the found row to the pivot_row position
            mat[pivot_row], mat[found] = mat[found], mat[pivot_row]
            
            # Scale pivot_row such that leading entry is 1
            pivot_val = mat[pivot_row][col]
            mat[pivot_row] = [x / pivot_val for x in mat[pivot_row]]

            # Eliminate the column in all other rows above and below
            for row in range(m):
                if row == pivot_row:
                    continue
                factor = mat[row][col]
                mat[row] = [mat[row][k] - factor * mat[pivot_row][k] for k in range(n)]
            pivot_cols.append(col)
            pivot_row += 1    # Next pivot goes in the row below this 
            if pivot_row == m:
                break          # All rows are utilised
        return mat, pivot_cols

    def rank(self):
        return len(self._pivot_cols)

    def nullity(self):
        return self.matrix.shape[1] - self.rank()     
       
    # Checks if a row is all zeros (used in row_space to skip zero rows from RREF)
    @staticmethod
    def _is_zero_row(row, tol = 1e-9):
        return all(abs(x) < tol for x in row)

    # Extracts a full column as a Vector 
    def _extract_column(self, col_index):
        m, n = self.matrix.shape
        if col_index < 0 or col_index >= n:
            raise IndexError(f"Column index {col_index} out of range for shape {self.matrix.shape}.")
        return Vector([self.matrix[row][col_index] for row in range(m)])
    
    # Extracts the full row as a Vector
    def _extract_row(self, row_index):
        m, _ = self.matrix.shape                       
        if row_index < 0 or row_index >= m:
            raise IndexError(f"Row index {row_index} out of range for shape {self.matrix.shape}.")
        return Vector(self.matrix[row_index])
    
    # Column space — original columns of A at pivot positions
    def column_space(self):
        return [self._extract_column(col) for col in self._pivot_cols]
    
    # Row space — non zero rows of RREF
    def row_space(self):
        return [Vector(row) for row in self._rref              
                if not FundamentalSubspaces._is_zero_row(row)]

    
    # Null space — solutions to Ax = 0
    def null_space(self):
        m, n = self.matrix.shape                                         
        free_cols = [col for col in range(n) if col not in self._pivot_cols]
        if not free_cols:
            return []     # only trivial solution x = 0

        basis = []
        for free_col in free_cols:
            # Build one basis vector per free variable
            # Set this free variable = 1, all other free variables = 0
            basis_vector = [0.0] * n
            basis_vector[free_col] = 1.0
            # For each pivot row, read off the value in the free column
            # and back-substitute: x_pivot = -rref[pivot_row][free_col]
            for pivot_row, pivot_col in enumerate(self._pivot_cols):
                basis_vector[pivot_col] = -self._rref[pivot_row][free_col]
            basis.append(Vector(basis_vector))
        return basis

    # Left null space — null space of A transpose (solutions to A^T x = 0)
    def left_null_space(self):
        return FundamentalSubspaces(self.matrix.transpose()).null_space()
    
if __name__ == "__main__":
    def input_matrix(name):
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

    # Rank Nullity verification
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

    # Sanity check: every null space vector should satisfy Ax = 0
    print("\nNull Space sanity check (A @ n_i should be zero vector):")
    for i, v in enumerate(fs.null_space()):
        result = A.apply_transformation(v)
        is_zero = all(abs(x) < 1e-9 for x in result)
        print(f"  A @ n{i+1} = {result} → {'✓' if is_zero else '✗'}")

    print("\nLeft Null Space sanity check (A^T @ l_i should be zero vector):")
    At = A.transpose()
    for i, v in enumerate(fs.left_null_space()):
        result = At.apply_transformation(v)
        is_zero = all(abs(x) < 1e-9 for x in result)
        print(f"  A^T @ l{i+1} = {result} → {'✓' if is_zero else '✗'}")


        








        

