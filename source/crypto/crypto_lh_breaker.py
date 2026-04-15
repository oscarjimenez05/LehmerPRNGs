import time
from z3 import *


def z3_rotate_left(val, r):
    return RotateLeft(val, r)


def z3_mix_arx(states):
    s1, s2, s3, s4 = states[1], states[2], states[3], states[4]
    s1 = s1 + s2;
    s4 = s4 ^ s1;
    s4 = z3_rotate_left(s4, 24)
    s3 = s3 + s4;
    s2 = s2 ^ s3;
    s2 = z3_rotate_left(s2, 12)
    s1 = s1 + s2;
    s4 = s4 ^ s1;
    s4 = z3_rotate_left(s4, 8)
    s3 = s3 + s4;
    s2 = s2 ^ s3;
    s2 = z3_rotate_left(s2, 7)

    return s1 ^ s2 ^ s3 ^ s4


def z3_xorshift64_step(x):
    x = x ^ (x << 13)
    x = x ^ LShR(x, 7)
    x = x ^ (x << 17)
    return x


def attempt_break():
    print("--- Attacking CryptoLehmer (ARX Version) ---")
    W, DELTA, SEQ_LEN = 6, 1, 2

    observed_ranks = [
        [0, 1, 2, 3, 4, 5],  # Window 1 is sorted
        [1, 2, 3, 4, 5, 0],  # Window 2
        [2, 3, 4, 5, 0, 1],  # Window 3
        [3, 4, 5, 0, 1, 2]  # Window 4
    ]

    print(f"Goal: Find 5 seeds that produce this ARX-mixed rank sequence.")

    solver = Solver()
    unknown_states = [BitVec(f's_{i}', 64) for i in range(5)]
    current_sym = list(unknown_states)
    window = []

    for _ in range(W):
        for j in range(5): current_sym[j] = z3_xorshift64_step(current_sym[j])
        window.append(z3_mix_arx(current_sym))

    start_time = time.time()

    # Constrain Sequence
    for step in range(SEQ_LEN):
        ranks = observed_ranks[step]
        # Add N^2 inequalities
        for i in range(W):
            for j in range(i + 1, W):
                if ranks[i] > ranks[j]:
                    solver.add(UGT(window[i], window[j]))
                else:
                    solver.add(ULT(window[i], window[j]))

        # Slide Window
        if step < SEQ_LEN - 1:
            for j in range(5): current_sym[j] = z3_xorshift64_step(current_sym[j])
            new_val = z3_mix_arx(current_sym)
            window = window[1:] + [new_val]

    print("Building complete. Running solver (Expect Timeout)...")
    solver.set("timeout", 30000) # 30 seconds
    result = solver.check()
    end_time = time.time()
    print(f"Finished in {end_time - start_time:.5f} seconds.")
    if result == sat:
        print("BROKEN! (This implies ARX failed)")
    elif result == unknown:
        print("TIMEOUT: Solver could not penetrate the ARX mixing.")
    else:
        print("UNSAT")


if __name__ == "__main__":
    attempt_break()