from typing import List

import pytest

from zeroband.utils.nibble_utils import read_nibbles


def pack_integers(integers: List[int], n_bits: int) -> bytes:
    """
    Packs a list of integers (each fitting in n_bits) into a byte array by concatenating their bits.
    If the total bits is not a multiple of 8, the final byte is padded with zeros in the LSB positions.
    """
    carry = 0
    carry_bits = 0
    byte_list = []
    for num in integers:
        if num >= (1 << n_bits):
            raise ValueError(f"Integer {num} does not fit in {n_bits} bits")
        carry = (carry << n_bits) | num
        carry_bits += n_bits
        while carry_bits >= 8:
            byte = carry >> (carry_bits - 8)
            byte_list.append(byte)
            carry = carry & ((1 << (carry_bits - 8)) - 1)
            carry_bits -= 8
    if carry_bits > 0:
        # Pad the remaining bits (on the right) with zeros to complete the last byte.
        byte = carry << (8 - carry_bits)
        byte_list.append(byte)
    return bytes(byte_list)


def test_read_nibbles_8_bits():
    # Test a clean case where each integer occupies exactly one byte.
    n_bits = 8
    source = [10, 20, 30, 40]
    data = pack_integers(source, n_bits)
    # Process the entire byte array at once.
    result, carry, carry_bits = read_nibbles(data, n_bits)
    assert result == source
    assert carry_bits == 0  # no leftover bits


def test_read_nibbles_4_bits_single_chunk():
    # Test reading 4-bit integers from a single complete chunk.
    n_bits = 4
    source = [1, 2, 3, 4, 5, 6, 7, 8]
    data = pack_integers(source, n_bits)
    result, carry, carry_bits = read_nibbles(data, n_bits)
    assert result == source
    assert carry_bits == 0


def test_read_nibbles_4_bits_multiple_chunks():
    # Test reading 4-bit integers when the byte stream is split across chunks.
    n_bits = 4
    source = [1, 2, 3, 4, 5, 6, 7, 8]
    data = pack_integers(source, n_bits)
    # Split data into two chunks (e.g., splitting on byte boundaries)
    mid = len(data) // 2
    result1, carry, carry_bits = read_nibbles(data[:mid], n_bits)
    result2, carry, carry_bits = read_nibbles(data[mid:], n_bits, carry, carry_bits)
    combined = result1 + result2
    assert combined == source
    assert carry_bits == 0


def test_read_nibbles_3_bits_single_chunk():
    # Test reading 3-bit integers. Note: values must be in range 0-7.
    n_bits = 3
    source = [1, 2, 3, 4, 5]
    data = pack_integers(source, n_bits)
    result, carry, carry_bits = read_nibbles(data, n_bits)
    assert result == source
    # The leftover bits (if any) should be less than n_bits.
    assert carry_bits < n_bits


def test_read_nibbles_stateful_across_chunks():
    # Test stateful reading with a non-clean size (n_bits = 5) across arbitrary chunks.
    n_bits = 5
    source = [5, 10, 15, 20, 25, 30]  # valid 5-bit integers (range 0-31)
    data = pack_integers(source, n_bits)
    chunk_size = 2  # split the data arbitrarily into chunks of 2 bytes
    result = []
    carry = 0
    carry_bits = 0
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        res, carry, carry_bits = read_nibbles(chunk, n_bits, carry, carry_bits)
        result.extend(res)
    assert result == source
    # Ensure any remaining carry does not form a full integer.
    assert carry_bits < n_bits

    # Uncomment the following line to run the tests when executing this script directly.


if __name__ == '__main__':
    pytest.main()
