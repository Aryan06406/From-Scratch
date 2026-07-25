""" 
m04_projection.py 

Implementation of vector projection operations from scratch. 

Topics covered 
-------------- 
- Scalar projection 
- Vector projection 
- Vector rejection 
- Orthogonal component 
- Projection matrix 
"""

from __future__ import annotations
from .m01_vector_ops import Vector
from .m03_linear_transformation import LinearTransformation

class Projection:
    # Verifies that both inputs are valid vectors for projection. 
    # Conditions: 
    # - Both operands must be Vector instances. 
    # - Both vectors must have the same dimension. 
    # - The vector being projected onto must be non-zero.
    @staticmethod
    def validate_projetion_vectors(u: Vector, v: Vector):
        Projection._validate_projcetion_vectors(u, v)

    # Internal helper
    # Internal helper for validating projection operands.
    @staticmethod
    def _validate_projcetion_vectors(u: Vector, v: Vector):
        if not isinstance(u, Vector):
            raise TypeError("The given input is not a vector.")
        if not isinstance(v, Vector):
            raise TypeError("The given input is not a vector.")
        u.check_dimension(v)
        if abs(v.dot_product(v)) < 1e-9:
            raise ValueError("Cannot project onto the zero vector.")
    
    # Computes the scalar projection of vector u onto vector v: comp_v(u) = (u · v) / ||v||
    @staticmethod
    def scalar_projection(u: Vector, v: Vector) -> float:
        Projection._validate_projcetion_vectors(u, v)
        return u.dot_product(v) / v.magnitude()

    # Computes the vector projection of u onto v: proj_v(u) = ((u · v) / (v · v)) v
    @staticmethod
    def vector_projection(u: Vector, v: Vector) -> Vector:
        Projection._validate_projcetion_vectors(u, v)
        scalar = u.dot_product(v) / v.dot_product(v)
        return scalar * v
    
    # Computes the rejection of u from v: rej_v(u) = u − proj_v(u)
    @staticmethod
    def rejection(u: Vector, v: Vector) -> Vector:
        return u - Projection.vector_projection(u, v)
    
    # Computes the component of u orthogonal to v. Equivalent to the rejection vector.
    @staticmethod
    def orthogonal_component(u: Vector, v: Vector) -> Vector:
        return Projection.rejection(u, v)
    
    # Computes the projection matrix onto the one-dimensional 
    # subspace spanned by vector v.
    # P = (v vᵀ) / (vᵀv)
    @staticmethod
    def projection_matrix(v: Vector) -> LinearTransformation:
        # The denominator is ||v||².
        denominator = v.dot_product(v)
        if abs(denominator) < 1e-9:
            raise ValueError("Zero vector has no projection matrix.")
        n = len(v)
        matrix = []
        # Construct the outer product vvᵀ
        for i in range(n):
            new_row = []
            for j in range(n):
                # Each entry is vi * vj.
                new_row.append(v[i] * v[j])
            matrix.append(new_row)
        # Scale the outer product by 1/(vᵀv) to obtain the projection matrix.
        for i in range(n):
            for j in range(n):
                matrix[i][j] /= denominator
        return LinearTransformation(matrix) 

if __name__ == "__main__":
    n1 = int(input("Enter the dimension of vector 1: "))
    v1 = []
    print("Enter the elements of vector 1: ")
    for i in range(n1):
        v1.append(float(input(f"Element {i+1}: ")))

    # Input for vector 2
    n2 = int(input("Enter the dimension of vector 2: "))
    v2 = []
    print("Enter the elements of vector 2: ")
    for i in range(n2):
        v2.append(float(input(f"Element {i+1}: ")))

    # Construct Vector objects from the user inputs.
    vector1 = Vector(v1)
    vector2 = Vector(v2)    

    # Scalar projection
    print("\nScalar Projection of vector 1 onto vector 2 is", Projection.scalar_projection(vector1, vector2))

    # Vector projection
    print("\nVector Projection of vector 1 onto vector 2 is", Projection.vector_projection(vector1, vector2))

    # Rejection
    print("\nRejection vector is", Projection.rejection(vector1, vector2))

    # Orthogonal component
    print("\nOrthogonal component of the vector projection is", Projection.orthogonal_component(vector1, vector2))

    # Projection matrix onto vector 1.
    print("\nProjection matrix:")
    print(Projection.projection_matrix(vector1))