"""
m01_vector_ops.py

Implementation of fundamental vector operations from scratch.

Topics covered
--------------
- Vector arithmetic
- Euclidean norm
- Dot product
- Cross product
- Hadamard product
- Unit vectors
- Angle between vectors
- Orthogonality
- Parallelism
- Distance
"""

from __future__ import annotations
import math 
from numbers import Real 

class Vector:
    __array_ufunc__ = None      # makes return NotImplemented -> Python calls Vector.__rmul__

    # Initialise a vector
    def __init__(self, elements: list[float] | tuple[float, ...]) -> None:
        elems = list(elements)
        for x in elems:
            if not isinstance(x, Real):
                raise TypeError(f"Vector elements must be real numbers, got {type(x).__name__}.")
        self.elements = elems

    # Returns a string representation of the vector
    def __str__(self) -> str:
        return str(self.elements)
    
    # Official representation used for debugging and interactive sessions.
    def __repr__(self) -> str:
        return f"Vector({self.elements})"

    # Returns the number of elements (dimensions) of the vector
    def __len__(self) -> int:
        return len(self.elements)
    
    # Allows iteration: for x in vector:
    def __iter__(self):
        return iter(self.elements)
    
    # Enables indexing: vector[i]
    def __getitem__(self, index):
        return self.elements[index]

    # Ensures both operands are vectors of equal dimension.
    # Raises an exception otherwise.
    def check_dimension(self, other: "Vector") -> None:
        self._check_dimension(other)

    def _check_dimension(self, other: "Vector") -> None:
        if not isinstance(other, Vector):
            raise TypeError("Operand must be vector.")
        if len(self) != len(other):
            raise ValueError(f"Dimension mismatch ({len(self)} != {len(other)})")   
        
    # Equal vectors
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self.elements == other.elements
    
    # Vector addition
    def __add__(self, other: "Vector") -> "Vector":
        self._check_dimension(other)
        return Vector([a + b for a, b in zip(self, other)])
    
    # Vector subtraction
    def __sub__(self, other: "Vector") -> "Vector": 
        self._check_dimension(other)
        return Vector([a - b for a, b in zip(self, other)])
    
    # Scalar multiplication: αv 
    def __mul__(self, scalar: int | float) -> "Vector":
        if not isinstance(scalar, (int, Real)):
            return NotImplemented
        return Vector([a * scalar for a in self.elements])
    
    def __rmul__(self, scalar: int | Real) -> "Vector":
        return self.__mul__(scalar)
    
    # Scalar division
    # True division: v / α
    def __truediv__(self, scalar: int | Real) -> "Vector":
        if not isinstance(scalar, (int, Real)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return Vector([y / scalar for y in self.elements]) 
    
    # Floor division (round offs to the closest value)
    def __floordiv__(self, scalar: int | Real) -> "Vector":
        if not isinstance(scalar, (int, Real)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return Vector([y // scalar for y in self.elements])
    
    # Computes the inner dot product: a . b = Σ(ai * bi)
    def dot_product(self, other: "Vector") -> float:
        self._check_dimension(other)
        return sum(a * b for a, b in zip(self, other)) 
    
    # Computes the cross product. Returns a vector orthogonal to both operands.
    # Defined only for R³.
    def cross_product(self, other: "Vector") -> "Vector":
        self._check_dimension(other)
        if len(self) != 3 or len(other) != 3:
            raise ValueError("Cross Product is defined for only 3D vectors")
        a1, a2, a3 = self.elements
        b1, b2, b3 = other.elements
        return Vector([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1])
    
    # Euclidean norm (L2 norm) also same as magnitude: ||v|| = √(Σ vi²)
    def magnitude(self) -> float:
        return math.hypot(*self.elements)   # overflow/underflow-safe
    
    # Negation of vector
    def __neg__(self) -> "Vector":
        return Vector([-x for x in self])   
    
    # Unit vector (Normalizes the vector to unit length): v̂ = v / ||v||
    def unit_vector(self) -> "Vector":
        mag = self.magnitude()
        if mag == 0:
            raise ZeroDivisionError("Zero vector doesn't have a defined unit vector.")
        return self / mag
    
    # Computes the angle between two vectors (degrees).
    # Uses Kahan's formula  theta = 2*atan2(||a^ - b^||, ||a^ + b^||)  on the unit vectors,
    # which stays accurate for tiny angles and near-antiparallel vectors (acos does not).
    def angle_between_vectors(self, other: "Vector") -> float:
        self._check_dimension(other)
        ma, mb = self.magnitude(), other.magnitude()
        if ma == 0 or mb == 0:
            raise ZeroDivisionError("Angle with zero vector is undefined.")
        if not (math.isfinite(ma) and math.isfinite(mb)):
            raise ValueError("Angle is undefined for NaN/inf components.")
        a, b = self / ma, other / mb
        return math.degrees(2.0 * math.atan2((a - b).magnitude(), (a + b).magnitude()))
    
    # Computes the Hadamard (element wise) product. 
    def hadamard_product(self, other: "Vector") -> "Vector":
        self._check_dimension(other)
        return Vector([x * y for x, y in zip(self, other)])
    
    # Distance between 2 vectors ie Euclidean distance: ||a - b||
    def distance_between_vectors(self, other: "Vector") -> float:
        return (self-other).magnitude()
    
    # Checks whether the vectors are orthogonal to each other or not
    # Scale-invariant: compares cos(theta) (dot of unit vectors) with a relative tolerance.
    def is_orthogonal(self, other: "Vector", tol: float = 1e-9) -> bool:
        self._check_dimension(other)
        ma, mb = self.magnitude(), other.magnitude()
        if ma == 0 or mb == 0:
            return True          # zero vector is orthogonal to everything
        return abs((self / ma).dot_product(other / mb)) <= tol
    
    # Checks whether the vectors are parallel or not (tolerant of floating point rounding).
    # Parallel  <=>  unit vectors coincide (a^ = b^) or are opposite (a^ = -b^).
    def is_parallel(self, other: "Vector", tol: float = 1e-12) -> bool:
        self._check_dimension(other)
        ma, mb = self.magnitude(), other.magnitude()
        if ma == 0 or mb == 0:
            return True          # zero vector is parallel to everything (0 * v = 0)
        a, b = self / ma, other / mb
        return min((a - b).magnitude(), (a + b).magnitude()) <= tol

    # Normalise vectors
    def normalize(self):
        return self.unit_vector()

if __name__ == "__main__":
    # Input for vector 1
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

    # Print both the vectors
    print("\nVector 1:", vector1)
    print("Vector 2:", vector2)
    
    # Addition
    print("\nAddition of vector 1 & 2:", vector1 + vector2)

    # Subtraction
    print("\nSubtraction of vector 1 & 2:", vector1 - vector2)

    # Equality of vectors
    print("\nAre vector 1 & 2 equal?:", vector1 == vector2)

    # Distance between 2 vectors
    print(f"\nDistance between the 2 vectors is {vector1.distance_between_vectors(vector2)} units.")

    # Scalar operations
    x = float(input("\nEnter scalar: "))
    print("\nScalar Multiplication")
    print(f"{x} * Vector 1 =", x * vector1)
    print(f"{x} * Vector 2 =", x * vector2)

    print("\nScalar Division (Standard Division)")
    print(f"Vector 1 / {x}=", vector1 / x)
    print(f"Vector 2 / {x} =", vector2 / x)

    print("\nScalar Division (Floor Division)") 
    print(f"Vector 1 // {x}=", vector1 // x)
    print(f"Vector 2 // {x}=", vector2 // x)

    # Dot product
    print("\nDot Product of vector 1 & 2:", vector1.dot_product(vector2))

    # Vector product
    try:
        print("\nVector Product of vector 1 & 2:", vector1.cross_product(vector2))
    except ValueError as e:
        print("\nVector Product:", e)

    # Element wise multiplication (Hadamard product)
    try:
        print("\nHadamard Product of vector 1 & 2:", vector1.hadamard_product(vector2))
    except ValueError as e:
        print("\nHadamard Product:", e)    

    # Magnitude
    print("\n Magnitude of vector 1:", vector1.magnitude())
    print("Magnitude of vector 2:", vector2.magnitude())

    # Unit vector
    print("\nUnit vector of vector 1:", vector1.unit_vector())
    print("Unit vector of vector 2:", vector2.unit_vector())

    # Angles between vectors
    print("\nAngle between two vectors:", vector1.angle_between_vectors(vector2))
    print("Are the vectors 1 & 2 orthogonal?:", vector1.is_orthogonal(vector2))

    # Negation of vectors
    print("\nNegation of vector 1:", -vector1)
    print("Negation of vector 2:", -vector2)

    # Parallel vectors
    print("\nAre vectors 1 & 2 parallel:", vector1.is_parallel(vector2))