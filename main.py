from hamiltonians import FreeWilson2D

test = FreeWilson2D(2,2,1,1)
print(test.field(0,0, False))
print(test.field(0,0, True))
h = test.hamiltonian()
print(h)
print(h.simplify())