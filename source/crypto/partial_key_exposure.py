import time
from z3 import *


def z3_rotate_left(val, r):
    return RotateLeft(val, r)


def z3_mix_arx(states):
    # states[1]..states[4] are used
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


def py_rot(n, r):
    return ((n << r) | (n >> (64 - r))) & 0xFFFFFFFFFFFFFFFF


def py_mix(s):
    s1, s2, s3, s4 = s[1], s[2], s[3], s[4]
    s1 = (s1 + s2) & 0xFFFFFFFFFFFFFFFF;
    s4 ^= s1;
    s4 = py_rot(s4, 24)
    s3 = (s3 + s4) & 0xFFFFFFFFFFFFFFFF;
    s2 ^= s3;
    s2 = py_rot(s2, 12)
    s1 = (s1 + s2) & 0xFFFFFFFFFFFFFFFF;
    s4 ^= s1;
    s4 = py_rot(s4, 8)
    s3 = (s3 + s4) & 0xFFFFFFFFFFFFFFFF;
    s2 ^= s3;
    s2 = py_rot(s2, 7)
    return s1 ^ s2 ^ s3 ^ s4


def py_xor(x):
    x = (x ^ (x << 13)) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 7)) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x << 17)) & 0xFFFFFFFFFFFFFFFF
    return x


def partial_exposure_attack():
    print("--- Partial Key Exposure Attack (CryptoLehmer) ---")
    print("Scenario: Attacker knows S0, S2, S3, S4.")

    W = 6
    DELTA = 14
    SEQ_LEN = 4

    secret_states = [12345, 67890, 11111, 22222, 33333]
    HIDDEN_INDEX = 1

    print(f"Secret States: {secret_states}")
    print(f"Hiding S{HIDDEN_INDEX} ({secret_states[HIDDEN_INDEX]}). Leaking the rest.")

    current_concrete = list(secret_states)
    window = []

    # Init
    for _ in range(W):
        for j in range(5): current_concrete[j] = py_xor(current_concrete[j])
        window.append(py_mix(current_concrete))

    observed_ranks = []

    for step in range(SEQ_LEN):
        ranks = [0] * W
        for i in range(W):
            count = 0
            for j in range(W):
                if window[j] < window[i]:
                    count += 1
                elif window[j] == window[i] and j < i:
                    count += 1
            ranks[i] = count
        observed_ranks.append(ranks)

        # Slide
        for j in range(5): current_concrete[j] = py_xor(current_concrete[j])
        new_val = py_mix(current_concrete)
        window = window[1:] + [new_val]

    print("Target ranks captured.")

    solver = Solver()
    sym_states = [None] * 5

    # Fix S0, S2, S3, S4 to their known values
    for i in range(5):
        if i == HIDDEN_INDEX:
            sym_states[i] = BitVec(f's_{i}', 64)  # unknown
        else:
            sym_states[i] = BitVecVal(secret_states[i], 64)  # known

    print("Constraints set. Solver has 4/5 of the key.")

    current_sym = list(sym_states)
    sym_window = []

    for _ in range(W):
        for j in range(5): current_sym[j] = z3_xorshift64_step(current_sym[j])
        sym_window.append(z3_mix_arx(current_sym))

    # Add Constraints
    print("Building equations...")
    for step in range(SEQ_LEN):
        target = observed_ranks[step]
        for i in range(W):
            for j in range(i + 1, W):
                if target[i] > target[j]:
                    solver.add(UGT(sym_window[i], sym_window[j]))
                else:
                    solver.add(ULT(sym_window[i], sym_window[j]))

        # Slide
        if step < SEQ_LEN - 1:
            for j in range(5): current_sym[j] = z3_xorshift64_step(current_sym[j])
            new_val = z3_mix_arx(current_sym)
            sym_window = sym_window[1:] + [new_val]

    print("--- Running Solver ---")
    start = time.time()
    result = solver.check()
    end = time.time()

    if result == sat:
        print(f"BROKEN in {end - start:.4f}s")
        m = solver.model()
        recovered = m[sym_states[HIDDEN_INDEX]].as_long()
        print(f"Recovered S{HIDDEN_INDEX}: {recovered}")
        print(f"Actual    S{HIDDEN_INDEX}: {secret_states[HIDDEN_INDEX]}")
        if recovered == secret_states[HIDDEN_INDEX]:
            print(">> EXACT MATCH (mixing function is reversible with partial info)")
        else:
            print(">> COLLISION (Found a valid alternative)")
    else:
        print("UNSAT or Timeout")


if __name__ == "__main__":
    partial_exposure_attack()