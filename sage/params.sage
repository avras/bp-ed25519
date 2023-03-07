#!/usr/bin/env sage

prime = 2^(255)-19
print("fe25519 parameters")
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

print("\n=============\nSanity checks\n=============")
print("MULTIPLICATIVE_GENERATOR^(MODULUS-1) =", gen^(prime-1))
print("TWO * TWO_INV =", two*two_inv)
print("ROOT_OF_UNITY^(2^S) =", root_of_unity^(2^s))
print("ROOT_OF_UNITY * ROOT_OF_UNITY_INV =", root_of_unity*root_of_unity_inv)
print("DELTA^t =", delta^t)
