"""
m02_norms.py

Implementation of common vector norms from scratch.

Topics covered
--------------
- Manhattan norm (L1)
- Euclidean norm (L2)
- Infinity norm (L∞)
- General p-norm
"""

from m01_vector_ops import Vector

class Norms:
    # Initializes the norm object for a Vector instance.
    def __init__(self, vector: Vector) -> None:
        if not isinstance(vector, Vector):
            raise TypeError("Expected a Vector object.")
        self.vector = vector
    
    # Computes the Manhattan (L1) norm: ||x||₁ = Σ |xi|
    def l1_norm(self) -> float:
        total = 0
        for x in self.vector:
            total += abs(x)
        return total
    
    # Computes the Euclidean (L2) norm: ||x||₂ = √(Σ xi²)
    def l2_norm(self) -> float:
        return self.vector.magnitude()
    
    # Computes the Infinity (Maximum) norm: ||x||∞ = max |xi|
    def infinity_norm(self) -> float:
        # The norm of the empty vector is defined as 0.
        if len(self.vector) == 0:   
            return 0
        maximum = abs(self.vector[0]) 
        for x in self.vector[1:]:
            if abs(x) > maximum:
                maximum = abs(x)
        return maximum               
    
    # Computes the general p-norm: ||x||p = (Σ |xi|ᵖ)^(1/p), where p > 0
    def p_norm(self, p: int | float) -> float:
        if p <= 0:
            raise ValueError("p must be greater than zero.")
        total = 0
        for x in self.vector:
            total += abs(x) ** p
        return total ** (1 / p)
    
if __name__ == "__main__":
    n = int(input("Enter the dimension of vector: "))
    v = []
    print("Enter the elements of vector 1: ")
    for i in range(n):
        v.append(float(input(f"Element {i+1}: ")))

    vector = Vector(v)
    norms = Norms(vector)

    print("Vector:", vector)
    print("L1 Norm:", norms.l1_norm())
    print("L2 Norm:", norms.l2_norm())
    print("L3 Norm:", norms.p_norm(3))
    print("Infinity Norm:", norms.infinity_norm())