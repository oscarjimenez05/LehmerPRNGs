import time
from z3 import *


def z3_xorshift64_step(x):
    x = x ^ (x << 13)
    x = x ^ LShR(x, 7)
    x = x ^ (x << 17)
    return x

def py_xor(x):
    x = (x ^ (x << 13)) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 7)) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x << 17)) & 0xFFFFFFFFFFFFFFFF
    return x


def get_ranks(window):
    # Calculate the relative ranks
    w = len(window)
    ranks = [0] * w
    for i in range(w):
        count = 0
        for j in range(w):
            if window[j] < window[i]:
                count += 1
            elif window[j] == window[i] and j < i:
                count += 1
        ranks[i] = count
    return ranks


def hunt_xor_shadow():
    print("--- Hunting for 'Shadow Seeds' in XorLehmer ---")

    # PARAMETERS
    W = 6
    DELTA = 1
    SEQ_LEN = 3
    SECRET_SEED = 123456789

    print(f"Target Secret Seed: {SECRET_SEED}")
    print(f"Generating target sequence of length {SEQ_LEN}...")

    current_state = SECRET_SEED
    window_concrete = []

    for _ in range(W):
        current_state = py_xor(current_state)
        window_concrete.append(current_state)

    observed_ranks = []

    sim_state = current_state
    sim_window = list(window_concrete)

    for step in range(SEQ_LEN):
        ranks = get_ranks(sim_window)
        observed_ranks.append(ranks)

        sim_state = py_xor(sim_state)
        sim_window = sim_window[DELTA:] + [sim_state]

    print("Target Ranks captured.")

    solver = Solver()

    unknown_seed = BitVec('seed', 64)

    current_sym = unknown_seed
    sym_window = []

    for _ in range(W):
        current_sym = z3_xorshift64_step(current_sym)
        sym_window.append(current_sym)

    # add constraints
    print("Building Z3 constraints...")

    for step in range(SEQ_LEN):
        target = observed_ranks[step]

        for i in range(W):
            for j in range(i + 1, W):
                if target[i] > target[j]:
                    solver.add(UGT(sym_window[i], sym_window[j]))
                else:
                    solver.add(ULT(sym_window[i], sym_window[j]))

        if step < SEQ_LEN - 1:
            current_sym = z3_xorshift64_step(current_sym)
            sym_window = sym_window[1:] + [current_sym]

    solver.add(unknown_seed != SECRET_SEED)

    print("\n--- Running Solver ---")
    print(f"Attempting to find a Shadow Seed for sequence length {SEQ_LEN}...")

    start_time = time.time()
    result = solver.check()
    end_time = time.time()

    if result == sat:
        print(f"\n[!] VULNERABILITY CONFIRMED in {end_time - start_time:.2f}s")
        print("Found a Shadow Seed! (Different state, same output)")
        m = solver.model()
        found_seed = m[unknown_seed].as_long()
        print(f"Original Seed: {SECRET_SEED}")
        print(f"Shadow Seed:   {found_seed}")
    else:
        print("\n[?] UNSAT")


if __name__ == "__main__":
    hunt_xor_shadow()