import time
from z3 import *


# --- 1. ARX Logic (The "Trapdoor") ---
def z3_rotate_left(val, r):
    return RotateLeft(val, r)


def z3_mix_arx(states):
    # Standard 1-Round ARX Logic
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


# --- 2. Concrete Helper (To generate the Target) ---
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


def hunt_shadow_seeds():
    print("--- Hunting for 'Shadow Seeds' in CryptoLehmer (ARX) ---")

    # PARAMETERS
    W = 6
    DELTA = 1
    SEQ_LEN = 3

    secret_states_concrete = [12345, 67890, 11111, 22222, 33333]
    print(f"Target Secret State: {secret_states_concrete}")

    print(f"Generating target sequence of length {SEQ_LEN}...")
    current_states = list(secret_states_concrete)
    window = []

    # Init Window
    for _ in range(W):
        for j in range(5): current_states[j] = py_xor(current_states[j])
        window.append(py_mix(current_states))

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

        # Slide Window
        for j in range(5): current_states[j] = py_xor(current_states[j])
        new_val = py_mix(current_states)
        window = window[1:] + [new_val]

    print("Target Ranks captured.")

    solver = Solver()

    sym_states = [BitVec(f's_{i}', 64) for i in range(5)]
    initial_sym_states = list(sym_states)  # Keep a reference to initial state

    current_sym = list(sym_states)
    sym_window = []

    for _ in range(W):
        for j in range(5): current_sym[j] = z3_xorshift64_step(current_sym[j])
        sym_window.append(z3_mix_arx(current_sym))

    print("Building Z3 constraints...")
    for step in range(SEQ_LEN):
        target = observed_ranks[step]

        # Force the output to match the target rank structure
        for i in range(W):
            for j in range(i + 1, W):
                if target[i] > target[j]:
                    solver.add(UGT(sym_window[i], sym_window[j]))
                else:
                    solver.add(ULT(sym_window[i], sym_window[j]))

        # Slide Window
        if step < SEQ_LEN - 1:
            for j in range(5): current_sym[j] = z3_xorshift64_step(current_sym[j])
            new_val = z3_mix_arx(current_sym)
            sym_window = sym_window[DELTA:] + [new_val]

    # We force at least one of the 5 state integers to be different.
    is_different = False
    for i in range(5):
        is_different = Or(is_different, initial_sym_states[i] != secret_states_concrete[i])

    solver.add(is_different)

    print("\n--- Running Solver ---")
    print(f"Attempting to find a COLLISION (Shadow Seed) for sequence length {SEQ_LEN}...")

    start_time = time.time()
    # Set timeout to 60 seconds
    solver.set("timeout", 60000)

    result = solver.check()
    end_time = time.time()

    if result == sat:
        print(f"\n[!] VULNERABILITY in {end_time - start_time:.2f}s")
        print("Found a Shadow Seed! (Different state, same output)")
        m = solver.model()
        for i in range(1,5):
            found = m[initial_sym_states[i]].as_long()
            print(f"Original S{i}: {secret_states_concrete[i]}")
            print(f"Shadow S{i}:   {found}")
    elif result == unknown:
        print(f"\n[+] SECURE (Timeout after {end_time - start_time:.2f}s)")
        print("Z3 could not find a shadow seed.")
    else:
        print("\n[+] UNSAT")


if __name__ == "__main__":
    hunt_shadow_seeds()