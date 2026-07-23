from m01_vector_ops import Vector
from m04_projection import Projection

class GramSchmidt:
    # Gram Schmidt is a procedure that strips away their mututal overlaps
    # (projections) and yields a set of mutually perpendicualr vectors
    # All the methods are static
    # Returns the component of v orthogonal to the current basis
    @staticmethod
    def orthogonalize_vector(v: Vector, basis: list[Vector]) -> Vector:
        if not isinstance(v, Vector):
            raise TypeError("v must be a vector.")
        x = Vector(list(v))
        for b in basis:
            x = x - Projection.vector_projection(x, b)
        return x

    # Applies gram schmidt to list of vectors and returns an orthogonal basis
    @staticmethod
    def orthogonal_basis(vectors: list[Vector]) -> list[Vector]:
        if not vectors:
            raise ValueError("Vector list cannot be empty.")
        basis = []
        for vector in vectors:
            if not isinstance(vector, Vector):
                raise TypeError("All elements must be Vector instances.")
            orthogonal = GramSchmidt.orthogonalize_vector(vector, basis)
            if orthogonal.magnitude() < 1e-9:
                continue
            basis.append(orthogonal)
        if not basis:
            raise ValueError("All vector lists are linearly dependent.") 
        return basis

    # Orthonormal basis means to normalise every vector to unit length in 
    # the orthogonal basis 
    @staticmethod
    def orthonormal_basis(vectors: list[Vector]) -> list[Vector]:
        orthogonal = GramSchmidt.orthogonal_basis(vectors)
        orthonormal = []
        for vector in orthogonal:
            orthonormal.append(vector.normalize()) 
        return orthonormal

    # Checks whether the list of vectors forms an orthogonal set or not 
    # Every distinct pair must have its dot product == 0
    @staticmethod
    def _is_orthogonal_basis(vectors: list[Vector]) -> bool:
        n = len(vectors)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if abs(vectors[i].dot_product(vectors[j])) > 1e-9:
                    return False
        return True

    # Checks whether the list of vetors forms an orthonormal set or not 
    # Must be orthogonal and every vector must have a unit magnitude
    @staticmethod
    def _is_orthonormal_basis(vectors: list[Vector]) -> bool:
        if not GramSchmidt._is_orthogonal_basis(vectors):
            return False
        for vector in vectors:
            if abs(vector.magnitude() - 1) > 1e-9:
                return False
        return True 

    # Verifies that the produced basis is valid, a report 
    @staticmethod
    def verify(original_vectors: list[Vector], basis: list[Vector]) -> None:
        print("Gram Schmidt Verification")
        
        # How many vectors survived (linear independence check)
        print(f"Input vectors: {len(original_vectors)}")
        print(f"Basis vectors kept: {len(basis)}")
        if len(basis) < len(original_vectors):
            dropped = len(original_vectors) - len(basis)
            print(f" → {dropped} vector(s) dropped (linearly dependent)")
        
        # Orthogonality check
        is_ortho = GramSchmidt._is_orthogonal_basis(basis)
        print(f"Orthogonal: {is_ortho}")
        
        # Unit length check
        is_unit = all(abs(v.magnitude() - 1.0) < 1e-9 for v in basis)
        print(f"Unit vectors: {is_unit}")
        if is_ortho and is_unit:
            print("Valid orthonormal basis.")
        elif is_ortho:
            print("Valid orthogonal basis (not normalized).")
        else:
            print("Basis failed orthogonality check.")
        
        # Print dot products between all pairs (should all be ≈ 0)
        print("\nPairwise dot products (should be ≈ 0):")
        n = len(basis)
        for i in range(n):
            for j in range(i + 1, n):
                dot = basis[i].dot_product(basis[j])
                print(f"  b{i+1} · b{j+1} = {dot:.2e}")

if __name__ == "__main__":
    def input_vector(label):
        n = int(input(f"Dimension of {label}: "))
        elements = list(map(float, input(f"Elements of {label}: ").split()))
        if len(elements) != n:
            raise ValueError("Incorrect number of elements.")
        return Vector(elements)

    k = int(input("How many vectors? "))
    vectors = [input_vector(f"vector {i+1}") for i in range(k)]

    print("\nOrthogonal Basis")
    ortho = GramSchmidt.orthogonal_basis(vectors)
    for i, v in enumerate(ortho):
        print(f"  b{i+1} = {v}")

    print("\nOrthonormal Basis")
    orthonormal = GramSchmidt.orthonormal_basis(vectors)
    for i, v in enumerate(orthonormal):
        print(f"  e{i+1} = {v}")

    print()
    GramSchmidt.verify(vectors, orthonormal)

    print("\nIs orthogonal basis:", GramSchmidt._is_orthogonal_basis(ortho))
    print("Is orthonormal basis:", GramSchmidt._is_orthonormal_basis(orthonormal))

