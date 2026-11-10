import random
import math
import secrets
from sys import stderr

# import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.random import Generator
from randomgen import PCG32


class PermCG32:

    def __init__(self, seed: int):
        self.generator = Generator(PCG32(seed=seed))


    def generate_chunk(self, n: int, debug):
        return self.generator.integers(0, 2**32, n, dtype='uint32')


def get_factorials(w):
    factorials = []
    for i in range(w):
        factorials.append(math.factorial(w - i - 1))
    return factorials


def lehmerize_sequence(sequence, n, minimum, maximum, w, delta, debug=0):
    r = maximum-minimum+1

    if delta == 0:
        delta = w

    factorials = get_factorials(w)
    R = math.factorial(w)
    thresh = R - (R%r)
    lehmer_codes = np.empty(shape=n, dtype=np.int64)

    # this will count the number of final numbers in the array
    count = 0

    if delta>w:
        print(f"Delta {delta} greater than window size {w}", stderr)
        exit(1)

    start = 0
    end = w

    seq_len = len(sequence)
    if seq_len < w:
        print(f"Sequence (len {seq_len}) is too short for window size {w}.", stderr)
        return np.array([], dtype=np.int64)  # Return empty array

    underl_sequence = sequence[:end]

    while count < n and end <= seq_len:
        # lehmer from scratch
        lehmer = 0
        is_initialized = 1
        for i in range(w-1):
            smaller = 0
            for j in range(i + 1, w):
                if underl_sequence[j] < underl_sequence[i]:
                    smaller += 1
            lehmer += smaller * factorials[i]

        if lehmer < thresh:
            lehmer_codes[count] = (lehmer%r) + minimum
            count += 1

        if debug:
            print(f"Base sequence: {underl_sequence}")
            print(f"Lehmer code: {lehmer} (valid? {lehmer<thresh})")
            print(f"Lehmer code adjusted for range: {(lehmer%r) + minimum})")
            print("\n----------\n")

        start += delta
        end += delta

        if end <= seq_len:
            underl_sequence = sequence[start:end]

    return lehmer_codes[:count]

def lcg(seed: int, n: int, a=1664525, c=1013904223, m=2 ** 32) -> [int]:
    """
    Default range is 0,2^32-1
    param seed (int): initial seed
    param n (int): number of elements to generate
    """
    result = np.empty(n)
    x = seed
    for i in range(n):
        x = (a * x + c) % m
        result[i] = x
    return result


def xorshift(seed, n) -> [int]:
    """
    Default range is 0,2^32-1
    param seed (int): initial seed
    param n (int): number of elements to generate
    """
    result = []
    x = seed
    for _ in range(n):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        result.append(x & 0xFFFFFFFF)
    return result


def _lehmer_from_ranks(rank_lists: [[int]]) -> [int]:
    n = len(rank_lists[0])
    factorials = [math.factorial(n - i - 1) for i in range(n)]
    results = []
    for ranks in rank_lists:
        n = len(ranks)
        code = 0
        for i in range(n):
            # below calculates normal lehmer code
            smaller = sum(1 for j in range(i + 1, n) if ranks[j] < ranks[i])
            code += smaller * factorials[i]
        results.append(code)
    return results


def lcg_lh(seed: int, n: int, w: int, a=1664525, c=1013904223, m=2 ** 32) -> [int]:
    """
    Overlapping lehmer code sliding window on top of LCG
    """
    return _lehmer_from_ranks(sliding_window_view(lcg(seed, n + w - 1, a, c, m), w))


def csprng(reps: int, max_exclusive: int):
    a_csprng = np.empty(reps)
    a_csprng = [secrets.randbelow(max_exclusive) for _ in range(reps)]
    return a_csprng


def mrs_tw(seed: int, reps: int, max_exclusive: int):
    bit_generator = np.random.MT19937(seed)
    rng = np.random.Generator(bit_generator)
    return rng.integers(0, max_exclusive, size=reps, dtype=np.uint32)


def pcg64(seed: int, reps: int, max_exclusive: int):
    rng = np.random.default_rng(seed)
    return rng.integers(0, max_exclusive, size=reps, dtype=np.uint32)


if __name__ == '__main__':
    # print(lcg_lh64(42, 10, 4, 2))
    normal_sequence = random_numbers_array = np.random.normal(loc=100, scale=15, size=1000000)
    normal_sequence = np.round(normal_sequence).astype(int)
    # plt.hist(normal_sequence)
    # plt.show()
    mini = 0
    maxi = 119
    w = 6
    lehmer = lehmerize_sequence(normal_sequence, 1000000, mini, maxi, w, 0, 0)
    print(min(lehmer))
    print(max(lehmer))
    # plt.hist(lehmer, bins=np.arange(mini -0.5, maxi+0.5, 1))
    # plt.show()
