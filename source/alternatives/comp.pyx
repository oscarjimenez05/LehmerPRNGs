# distutils: language=c
# cython: language_level=3

import numpy as np
cimport numpy as np
import cython
import math
from libc.stdint cimport uint64_t

np.import_array()

# -------------------------------------------------------------------------
# GENERATOR 1: Lm_LCG (Large Modulus LCG with proper discard)
# -------------------------------------------------------------------------

cdef class LmLcg:
    cdef uint64_t state
    cdef uint64_t a
    cdef uint64_t c
    cdef long long minimum
    cdef long long maximum
    cdef long long r
    cdef uint64_t thresh
    cdef bint needs_discard

    def __cinit__(self, uint64_t seed, long long minimum, long long maximum):
        """
        :param seed: initial state
        :param minimum: inclusive
        :param maximum: inclusive
        """
        self.state = seed
        self.a = 6364136223846793005
        self.c = 1442695040888963407

        self.minimum = minimum
        self.maximum = maximum
        self.r = self.maximum - self.minimum + 1

        # Determine exactly how many values complete a full cycle in the 2^64 range
        R = 2 ** 64
        remainder = R % self.r

        # If the range divides 2^64 perfectly, no discarding is needed
        self.needs_discard = (remainder != 0)

        if self.needs_discard:
            self.thresh = R - remainder
        else:
            self.thresh = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray generate_chunk(self, int n, int debug):
        cdef np.ndarray[np.uint64_t, ndim=1] results = np.empty(n, dtype=np.uint64)
        cdef int count = 0

        # PINNED LOCAL VARIABLES
        cdef uint64_t p_state = self.state
        cdef uint64_t p_thresh = self.thresh
        cdef uint64_t p_minimum = self.minimum
        cdef uint64_t p_r = self.r
        cdef uint64_t p_a = self.a
        cdef uint64_t p_c = self.c
        cdef bint p_needs_discard = self.needs_discard

        while count < n:
            # Advance state natively at 64 bits
            p_state = p_a * p_state + p_c

            # If the value falls within the bounds of completed cycles, accept it
            if not p_needs_discard or p_state < p_thresh:
                results[count] = (p_state % p_r) + p_minimum
                count += 1

            if debug:
                print(f"State: {p_state}")
                print(f"Valid? {not p_needs_discard or p_state < p_thresh}")
                print(f"Adjusted for range: {(p_state % p_r) + p_minimum}")
                print("\n----------\n")

        # CRUCIAL, update persistent state
        self.state = p_state

        return results

# -------------------------------------------------------------------------
# GENERATOR 2: HighBitsLcg (Output only highest 32 bits with proper discard)
# -------------------------------------------------------------------------

cdef class HighBitsLcg:
    cdef uint64_t state
    cdef uint64_t a
    cdef uint64_t c
    cdef long long minimum
    cdef long long maximum
    cdef long long r
    cdef uint64_t thresh

    def __cinit__(self, uint64_t seed, long long minimum, long long maximum):
        """
        :param seed: initial state
        :param minimum: inclusive
        :param maximum: inclusive
        """
        self.state = seed
        self.a = 6364136223846793005
        self.c = 1442695040888963407

        self.minimum = minimum
        self.maximum = maximum
        self.r = self.maximum - self.minimum + 1

        # We are outputting 32-bit values, so our base range is 2^32
        R = 2 ** 32
        remainder = R % self.r
        self.thresh = R - remainder

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray generate_chunk(self, int n, int debug):
        cdef np.ndarray[np.uint64_t, ndim=1] results = np.empty(n, dtype=np.uint64)
        cdef int count = 0
        cdef uint64_t raw

        # PINNED LOCAL VARIABLES
        cdef uint64_t p_state = self.state
        cdef uint64_t p_thresh = self.thresh
        cdef uint64_t p_minimum = self.minimum
        cdef uint64_t p_r = self.r
        cdef uint64_t p_a = self.a
        cdef uint64_t p_c = self.c

        while count < n:
            # Advance state natively at 64 bits
            p_state = p_a * p_state + p_c

            # Bitshift down to extract only the upper 32 bits
            raw = p_state >> 32

            # Discard to avoid modulo bias mapping 2^32 to our range `r`
            if raw < p_thresh:
                results[count] = (raw % p_r) + p_minimum
                count += 1

            if debug:
                print(f"State: {p_state}")
                print(f"Raw (top 32 bits): {raw}")
                print(f"Valid? {raw < p_thresh}")
                print(f"Adjusted for range: {(raw % p_r) + p_minimum}")
                print("\n----------\n")

        # CRUCIAL, update persistent state
        self.state = p_state

        return results