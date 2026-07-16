import math

class Norms:
    def __init__(self, values):
        self.values = values
    
    # Manhattan norm
    def l1_norm(self):
        total = 0
        for x in self.values:
            total += abs(x)
        return total
    
    # Euclidean norm 
    def l2_norm(self):
        total = 0
        for x in self.values:
            total += x * x
        return math.sqrt(total)
    
    # Maximum norm
    def infinity_norm(self):
        if len(self.values) == 0:
            return 0
        maximum = abs(self.values[0]) 
        for x in self.values[1:]:
            if abs(x) > maximum:
                maximum = abs(x)
        return maximum               
    
    # General p norm
    def p_norm(self, p):
        if p <= 0:
            raise ValueError("p must be greater than zero.")
        total = 0
        for x in self.values:
            total += abs(x) ** p
        return total ** (1 / p)
    
n = int(input("Enter the dimension of vector: "))
v = []
print("Enter the elements of vector 1: ")
for i in range(n):
    v.append(float(input(f"Element {i+1}: ")))

vector = Norms(v)

print("Vector:", vector.values)
print("L1 Norm:", vector.l1_norm())
print("L2 Norm:", vector.l2_norm())
print("L3 Norm:", vector.p_norm(3))
print("Infinity Norm:", vector.infinity_norm())    