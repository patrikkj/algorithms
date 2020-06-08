import random


def p(x):
    return str(x)[::-1].index("1")
print("\n"*10)
f = ['0'] * 64
print(f"f: {''.join(f)}\n")

while True:
    r = bin(random.getrandbits(64))
    i = p(r) + 1
    f[-i] = '1'
    print(f"num: {r}")
    print(f"idx: {i}")
    print(f"filter: {''.join(f)}")
    print("\n")
    input()
