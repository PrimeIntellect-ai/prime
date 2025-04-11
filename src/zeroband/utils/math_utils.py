import math


def next_common_multiple(a: int, b: int, x: int) -> int:
    lcm_val = abs(a * b) // math.gcd(a, b)
    return ((x // lcm_val) + 1) * lcm_val
