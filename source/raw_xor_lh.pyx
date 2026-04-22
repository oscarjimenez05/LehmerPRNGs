# distutils: language=c
# cython: language_level=3

import numpy as np
cimport numpy as np
import cython
import math
from libc.string cimport memmove
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint64_t

np.import_array()

cdef inline uint64_t xorshift64_step(uint64_t x) nogil:
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x

cdef class RawXorLehmer:
    cdef uint64_t state
    cdef uint64_t *window_buffer
    cdef uint64_t *factorials
    cdef int w
    cdef int delta
    cdef bint is_initialized

    def __cinit__(self, uint64_t seed, int w, int delta):
        if seed == 0: seed = 123456789
        self.state = seed
        self.w = w

        # fully non-overlapping
        if delta == 0:
            self.delta = w
        else:
            self.delta = delta

        self.is_initialized = 0

        self.window_buffer = <uint64_t *> malloc(w * sizeof(uint64_t))
        self.factorials = <uint64_t *> malloc(w * sizeof(uint64_t))

        cdef int i
        for i in range(w):
            self.factorials[i] = math.factorial(w - i - 1)

    def __dealloc__(self):
        if self.window_buffer: free(self.window_buffer)
        if self.factorials: free(self.factorials)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray generate_raw_chunk(self, int n):
        """Generates exactly n Lehmer codes without discarding or modulo."""
        cdef np.ndarray[np.uint64_t, ndim=1] results = np.empty(n, dtype=np.uint64)
        cdef int i, j, k, count
        cdef int smaller
        cdef uint64_t lehmer

        if not self.is_initialized:
            for i in range(self.w):
                self.state = xorshift64_step(self.state)
                self.window_buffer[i] = self.state
            self.is_initialized = 1

        # PINNED LOCAL VARIABLES
        cdef uint64_t p_state = self.state
        cdef int p_w = self.w
        cdef int p_delta = self.delta
        cdef uint64_t *p_window = self.window_buffer
        cdef uint64_t *p_factorials = self.factorials

        # No while loop, strict for-loop for n iterations
        for count in range(n):
            if p_delta < p_w:
                memmove(p_window,
                        p_window + p_delta,
                        (p_w - p_delta) * sizeof(uint64_t))

            # generate delta new numbers using Xorshift
            for k in range(p_w - p_delta, p_w):
                p_state = xorshift64_step(p_state)
                p_window[k] = p_state

            lehmer = 0
            for i in range(p_w):
                smaller = 0
                for j in range(i + 1, p_w):
                    smaller += (p_window[j] < p_window[i])
                lehmer += smaller * p_factorials[i]

            results[count] = lehmer

        self.state = p_state
        return results