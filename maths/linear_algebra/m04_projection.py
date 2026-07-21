from m01_vector_ops import Vector
from m03_linear_transformation import LinearTransformation

class Projection:
    # Projection is not a isolated topic but rather a knock off of combination of
    # several topics of vectors and linear transformation onto each other 
    # hence why we use @staticmethod. 
    # Validating the vectors
    def validate_projetion_vectors(u: Vector, v: Vector):
        Projection._validate_projetion_vectors(u, v)

    def _validate_projetion_vectors(u: Vector, v: Vector):
        if u is not Vector:
            raise TypeError("The given input is not a vector.")
        if v is not Vector:
            raise TypeError("The given input is not a vector.")
        u.check_dimension(v)
        if abs(v.dot_product(v)) < 1e-9:
            raise ValueError
    
    # Scalar projection of vector u onto vector v
    @staticmethod
    def scalar_projection(u: Vector, v: Vector) -> float:
        Projection._validate_projetion_vectors(u, v)
        return u.dot_product(v) / v.magnitude()

    # Vector projection of vector u onto vector v
    @staticmethod
    def vector_projection(u: Vector, v: Vector) -> Vector:
        Projection._validate_projetion_vectors(u, v)
        scalar = u.dot_product(v) / v.dot_product(v)
        return scalar * v
    
    # Rejection vector
    @staticmethod
    def rejection(u: Vector, v: Vector) -> Vector:
        return u - Projection.vector_projection(u, v)
    
    # Orthogonal component of the vector
    @staticmethod
    def orthogonal_component(u: Vector, v: Vector) -> Vector:
        return Projection.rejection(u, v)
    
    # Projection matrix
    @staticmethod
    def projection_matrix(v: Vector) -> LinearTransformation:
        denominator = v.dot_product(v)
        if abs(denominator) < 1e-9:
            raise ValueError("Zero vector has no projection matrix.")
        n = len(v)
        matrix = []
        for i in range(n):
            new_row = []
            for j in range(n):
                new_row.append(v[i] * v[j] / denominator)
            matrix.append(new_row)
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

    # Calling the class Vector
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

    # Projection matrix
    print("\nProjection matrix:")
    print(Projection.projection_matrix(vector1))