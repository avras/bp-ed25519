#!/usr/bin/env sage

prime = 2^(255)-19
print("ed25519 parameters")
print("MODULUS =", prime.hex())
print("NUM_BITS =", len(prime.bits()))
print("CAPACITY =", len(prime.bits())-1)
F = GF(prime)

two = F(2)
two_inv = two^(-1)
print("TWO_INV =", hex(two_inv))

gen = F.primitive_element()
print("MULTIPLICATIVE_GENERATOR =", gen)

s = 0
while is_even((prime-1)/(2^s)):
    s += 1
print("S =", s)

t = (prime-1)/(2^s)
root_of_unity = gen^t
print("ROOT_OF_UNITY =", hex(root_of_unity))

root_of_unity_inv = root_of_unity^(-1)
print("ROOT_OF_UNITY_INV =", hex(root_of_unity_inv))

delta = gen^(2^s)
print("DELTA = ", delta)

d = F(-121665) * F(121666)^(-1)
print("D =", d) # Check value in https://www.rfc-editor.org/rfc/rfc8032#section-5.1
print("D (hex) =", hex(d))

sqrt_minus_one = F(2)^((prime-1)/4)
print("SQRT_MINUS_ONE =", sqrt_minus_one)
print("SQRT_MINUS_ONE (hex) =", hex(sqrt_minus_one))

curve_order = 2^252+27742317777372353535851937790883648493
print("L =", d) # Check value in https://www.rfc-editor.org/rfc/rfc8032#section-5.1
print("L (hex) =", hex(curve_order))

print("\n=============\nSanity checks\n=============")
print("MULTIPLICATIVE_GENERATOR^(MODULUS-1) =", gen^(prime-1))
print("TWO * TWO_INV =", two*two_inv)
print("ROOT_OF_UNITY^(2^S) =", root_of_unity^(2^s))
print("ROOT_OF_UNITY * ROOT_OF_UNITY_INV =", root_of_unity*root_of_unity_inv)
print("DELTA^t =", delta^t)
print("S =", delta^t)
print("SQRT_MINUS_ONE * SQRT_MINUS_ONE + ONE =", (sqrt_minus_one*sqrt_minus_one + 1) % prime)
