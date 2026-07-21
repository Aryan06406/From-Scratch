import math 

class Vector:
    # Initialise a vector
    def __init__(self, elements):
        self.elements = list(elements)

    # Returns a string representation of the vector
    def __str__(self):
        return str(self.elements)
    
    #  Returns unambiguous string representation for debugging
    def __repr__(self):
        return f"Vector({self.elements})"

    # Returns the number of elements (dimensions) of the vector
    def __len__(self):
        return len(self.elements)
    
    # Returns an iterator over vector elements
    def __iter__(self):
        return iter(self.elements)
    
    # Returns the element at specified index 
    def __getitem__(self,index):
        return self.elements[index]

    # Checks whether the vectors are of same dimension or not 
    def check_dimension(self, other):
        self._check_dimension(other)

    def _check_dimension(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Operand must be vector.")
        if len(self) != len(other):
            raise ValueError(f"Dimension mismatch ({len(self)} != {len(other)})")   
        
    # Equal vectors
    def __eq__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return self.elements == other.elements
    
    # Vector addition
    def __add__(self, other):
        self._check_dimension(other)
        return Vector([a + b for a, b in zip(self, other)])
    
    # Vector subtraction
    def __sub__(self, other):
        self._check_dimension(other)
        return Vector([a - b for a, b in zip(self, other)])
    
    # Scalar multiplication 
    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector([a * scalar for a in self.elements])
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    # Scalar division
    # True division (no round off) 
    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return Vector([y / scalar for y in self.elements]) 
    
    # Floor division (round offs to the closest value)
    def __floordiv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return Vector([y // scalar for y in self.elements])
    
    # Dot product
    def dot_product(self, other):
        self._check_dimension(other)
        return sum(a * b for a, b in zip(self, other)) 
    
    # Vector/Cross product (strictly for 3D only)
    def cross_product(self, other):
        self._check_dimension(other)
        if len(self) != 3 or len(other) != 3:
            raise ValueError("Cross Product is defined for only 3D vectors")
        a1, a2, a3 = self.elements
        b1, b2, b3 = other.elements
        return Vector([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1])
    
    # Magnitude
    def magnitude(self):
        return math.sqrt(sum(x**2 for x in self.elements))  
    
    # Negation of vector
    def __neg__(self):
        return Vector([-x for x in self])   
    
    # Unit vector (vector normalisation)
    def unit_vector(self):
        mag = self.magnitude()
        if mag == 0:
            raise ZeroDivisionError("Zero vector doesn't have a defined unit vector.")
        return self / mag
    
    # Angle between vectors
    def angle_between_vectors(self, other):
        if self.magnitude() == 0 or other.magnitude() == 0:
            raise ZeroDivisionError("Angle with zero vector is undefined.")
        theta = ((self.dot(other)) / ((self.magnitude()) * (other.magnitude())))
        theta = max(-1.0, min(1.0, theta))
        return math.degrees(math.acos(theta)) 
    
    # Element wise multiplication (Hadamard product)
    def hadamard_product(self, other):
        self._check_dimension(other)
        return Vector([x * y for x, y in zip(self, other)])
    
    # Distance between 2 vectors
    def distance_between_vectors(self, other):
        return (self-other).magnitude()
    
    # Checks whether the vectors are orthogonal to each other or not
    def is_orthogonal(self, other):
        return abs(self.dot(other)) < 1e-9
    
    # Checks whether the vectors are parallel or not
    def is_parallel(self, other):
        self._check_dimension(other)
        ratio = None
        for a, b in zip(self, other):
            if a == 0 and b == 0:
                continue
            if a == 0 or b == 0:
                return False
            if ratio is None:
                ratio = (a, b)
            elif a * ratio[1] != b * ratio[0]:
                return False
        return True             

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