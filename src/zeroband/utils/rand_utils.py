import numba
from typing import Tuple


@numba.njit
def lsfr_rand_u64(lsr_seed, hi) -> Tuple[int, int]:
    """
    A 64-bit linear feedback shift register (LFSR) pseudorandom generator.

    :param lsr_seed: The current 64-bit seed (non-zero for max period).
    :param hi: the maximum value to be returned (exclusive)
    :return: A tuple (random_value, new_seed).
    """
    new_bit = ((lsr_seed >> 63) ^ (lsr_seed >> 62) ^ (lsr_seed >> 60) ^ (lsr_seed >> 59)) & 1
    new_seed = ((lsr_seed << 1) & 0xFFFFFFFFFFFFFFFF) | new_bit
    return new_seed % hi, new_seed