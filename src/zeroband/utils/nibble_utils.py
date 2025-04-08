from typing import Tuple, List

import numba
import numpy as np
from numba.typed import List as TypedList


@numba.jit
def read_nibbles_numba(arr: np.ndarray, n_bits: int, carry: int, carry_bits: int):
    """
    Reads integers of size n_bits from an array of bytes (uint8), interpreting the bit stream
    in big-endian order (MSB-first). That is, the first bit in the stream is the most significant
    bit of the first integer.

    Parameters:
      arr (np.ndarray): Array of uint8 values representing the byte stream.
      n_bits (int): Number of bits per integer.
      carry (int): Carry-over bits from previous chunks.
      carry_bits (int): Number of valid bits in carry.

    Returns:
      Tuple containing:
        - A typed list of integers read from the combined bit stream.
        - The updated carry.
        - The updated number of carry bits.
    """
    result = TypedList()
    for i in range(arr.shape[0]):
        # Shift the current carry left by 8 bits and append the new byte.
        carry = (carry << 8) | arr[i]
        carry_bits += 8

        # While we have enough bits for one integer, extract the most significant n_bits.
        while carry_bits >= n_bits:
            # Extract the integer from the leftmost (most significant) n_bits.
            value = carry >> (carry_bits - n_bits)
            result.append(value)
            # Remove the extracted bits from carry.
            carry = carry & ((1 << (carry_bits - n_bits)) - 1)
            carry_bits -= n_bits
    return result, carry, carry_bits


def read_nibbles(chunk_bytes: bytes, n_bits: int, carry: int = 0, carry_bits: int = 0) -> Tuple[List[int], int, int]:
    """
    Reads integers of size n_bits from a byte array while preserving state across chunks.
    Bits are processed in big-endian order (most significant bit first).

    :param chunk_bytes: The current chunk of bytes.
    :param n_bits: The number of bits per integer (n_bits <= 64).
    :param carry: Accumulated bits carried over from the previous chunk (default 0).
    :param carry_bits: The number of valid bits in carry (default 0).

    :returns:
     - A list of integers read from the combined carry and current chunk.
     - The updated carry (remaining bits not enough to form an integer).
     - The number of valid bits in the updated carry.
    """
    # Convert the byte array to a NumPy array for Numba processing.
    arr = np.frombuffer(chunk_bytes, dtype=np.uint8)
    result, carry, carry_bits = read_nibbles_numba(arr, n_bits, carry, carry_bits)
    return list(result), carry, carry_bits
