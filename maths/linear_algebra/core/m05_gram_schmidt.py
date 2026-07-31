""" 
m05_gram_schmidt.py 

Implementation of the Gram Schmidt orthogonalization process from scratch.

Topics covered 
-------------- 
- Orthogonalization of a vector 
- Orthogonal basis 
- Orthonormal basis 
- Orthogonality verification 
- Orthonormality verification 
"""

from __future__ import annotations
from maths.linear_algebra.core.m01_vector_ops import Vector
from maths.linear_algebra.core.m04_projection import Projection

class GramSchmidt:
    # Implements the Gram-Schmidt orthogonalization process. 
    # The algorithm removes the projection of each vector onto the
    # previously computed basis vectors, producing an orthogonal basis
    # spanning the same subspace. 
    # All methods are static because the algorithm maintains no state.
    @staticmethod
    # Computes the component of v orthogonal to the current basis. 
    # The projections of v onto every basis vector are removed,
    # leaving only the orthogonal component.
    def orthogonalize_vector(v: Vector, basis: list[Vector]) -> Vector:
        if not isinstance(v, Vector):
            raise TypeError("v must be a vector.")
        # Make a copy so the original vector remains unchanged.
        x = Vector(list(v))
        # Remove the projection onto each existing basis vector.
        for b in basis:
            x = x - Projection.vector_projection(x, b)
        return x

    # Applies the Gram Schmidt process to construct 
    # an orthogonal basis from a list of vectors.
    # Linearly dependent vectors are discarded.
    @staticmethod
    def orthogonal_basis(vectors: list[Vector]) -> list[Vector]:
        if not vectors:
            raise ValueError("Vector list cannot be empty.")
        basis = []
        for vector in vectors:
            if not isinstance(vector, Vector):
                raise TypeError("All elements must be Vector instances.")
            # Compute the component orthogonal to the current basis.
            orthogonal = GramSchmidt.orthogonalize_vector(vector, basis)
            # Ignore vectors that are linearly dependent.
            if orthogonal.magnitude() < 1e-9:
                continue
            basis.append(orthogonal)
        if not basis:
            raise ValueError("The input vectors are linearly dependent; no non zero basis vectors remain.") 
        return basis

    # Converts an orthogonal basis into an orthonormal basis
    # by normalizing every basis vector. @staticmethod 
    @staticmethod
    def orthonormal_basis(vectors: list[Vector]) -> list[Vector]:
        # Normalize each orthogonal basis vector.
        orthogonal = GramSchmidt.orthogonal_basis(vectors)
        orthonormal = []
        for vector in orthogonal:
            orthonormal.append(vector.normalize()) 
        return orthonormal

    # Checks whether a collection of vectors is orthogonal. 
    # Every distinct pair must satisfy: ui · uj = 0 & within numerical tolerance.
    @staticmethod
    def is_orthogonal_basis(vectors: list[Vector]) -> bool:
        n = len(vectors)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if abs(vectors[i].dot_product(vectors[j])) > 1e-9:
                    return False
        return True

    # Checks whether a collection of vectors is orthonormal. 
    # Conditions: The vectors are mutually orthogonal & Every vector has unit length.
    @staticmethod
    def is_orthonormal_basis(vectors: list[Vector]) -> bool:
        if not GramSchmidt.is_orthogonal_basis(vectors):
            return False
        for vector in vectors:
            if abs(vector.magnitude() - 1) > 1e-9:
                return False
        return True 

    # Prints a verification report for the computed basis. 
    # The report includes: Number of vectors retained, Orthogonality check,
    # Unit length check & Pairwise dot products
    @staticmethod
    def verify(original_vectors: list[Vector], basis: list[Vector]) -> None:
        print("Gram Schmidt Verification")
        
        # Compare the number of input vectors and basis vectors.
        print(f"Input vectors: {len(original_vectors)}")
        print(f"Basis vectors kept: {len(basis)}")
        if len(basis) < len(original_vectors):
            dropped = len(original_vectors) - len(basis)
            print(f" → {dropped} vector(s) dropped (linearly dependent)")
        
        # Check whether the basis vectors are mutually orthogonal.
        is_ortho = GramSchmidt.is_orthogonal_basis(basis)
        print(f"Orthogonal: {is_ortho}")
        
        # Verify that every basis vector has unit length.
        is_unit = all(abs(v.magnitude() - 1.0) < 1e-9 for v in basis)
        # Display every pairwise dot product.
        print(f"Unit vectors: {is_unit}")
        if is_ortho and is_unit:
            print("Valid orthonormal basis.")
        elif is_ortho:
            print("Valid orthogonal basis (not normalized).")
        else:
            print("Basis failed orthogonality check.")
        
        # These values should be approximately zero.
        print("\nPairwise dot products (should be ≈ 0):")
        n = len(basis)
        for i in range(n):
            for j in range(i + 1, n):
                dot = basis[i].dot_product(basis[j])
                print(f"  b{i+1} · b{j+1} = {dot:.2e}")

if __name__ == "__main__":
    # Reads a vector from standard input.
    def input_vector(label: str) -> Vector:
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

    print("\nIs orthogonal basis:", GramSchmidt.is_orthogonal_basis(ortho))
    print("Is orthonormal basis:", GramSchmidt.is_orthonormal_basis(orthonormal))

