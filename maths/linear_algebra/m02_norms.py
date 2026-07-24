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

import math
from m01_vector_ops import Vector

class Norms:
    # Initializes the norm object with a vector (or any sequence of numbers).
    def __init__(self, vector: "Vector") -> None:
        self.vector = vector
    
    # Computes the Manhattan (L1) norm: ||x||₁ = Σ |xi|
    def l1_norm(self) -> float:
        total = 0
        for x in self.vector:
            total += abs(x)
        return total
    
    # Computes the Euclidean (L2) norm: ||x||₂ = √(Σ xi²)
    def l2_norm(self) -> float:
        total = 0
        for x in self.vector:
            total += x * x
        return math.sqrt(total)
    
    # Computes the Infinity (Maximum) norm: ||x||∞ = max |xi|
    def infinity_norm(self) -> float:
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

    vector = Norms(v)

    print("Vector:", vector.vector)
    print("L1 Norm:", vector.l1_norm())
    print("L2 Norm:", vector.l2_norm())
    print("L3 Norm:", vector.p_norm(3))
    print("Infinity Norm:", vector.infinity_norm()) 