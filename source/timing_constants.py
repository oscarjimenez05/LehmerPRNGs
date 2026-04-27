import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import xor_lh as xlh
import raw_xor_lh as rxlh

def raw_xor_lh(seed, w, n):
    generator = rxlh.RawXorLehmer(seed, w, 0)
    return generator.generate_raw_chunk(n)


def generalized_xor_lh(seed, w, R, n):
    generator = xlh.XorLehmer(seed, w, 0, 0, R - 1)
    return generator.generate_chunk(n, 0)


# =====================================================================

def profile_base_cost(trials=10, n_generations=1_000_000):
    """
    Profile the raw Lehmer calculation cost.
    Uses multiple trials and takes the MINIMUM time to avoid OS/thermal noise.
    """
    print("--- Profiling Base Cost ---", file=sys.stderr)
    w_values = list(range(2, 19))
    times = []

    for w in w_values:
        best_time = float('inf')
        for _ in range(trials):
            # let laptop thermals settle
            time.sleep(0.5)

            start = time.perf_counter_ns()
            raw_xor_lh(12345, w, n_generations)
            end = time.perf_counter_ns()

            elapsed = (end - start) / 1e9  # convert to seconds
            if elapsed < best_time:
                best_time = elapsed

        time_per_gen = best_time / n_generations
        times.append(time_per_gen)
        print(f"w={w}: {time_per_gen:.2e} seconds per gen", file=sys.stderr)

    # fit a quadratic curve: t(w) = c1*w^2 + c2*w + c3
    coeffs = np.polyfit(w_values, times, 2)
    print(f"\nFitted Constants: c1={coeffs[0]:.2e}, c2={coeffs[1]:.2e}, c3={coeffs[2]:.2e}", file=sys.stderr)
    print(f"{coeffs[0]:.2e} {coeffs[1]:.2e} {coeffs[2]:.2e}")
    return coeffs, w_values, times


def expected_time(w, R, coeffs):
    """
    Mathematical Model.
    Calculates expected time to get a valid number based on acceptance probability.
    """
    w_fact = math.factorial(w)
    if w_fact < R:
        return float('inf')  # Invalid window size for this range

    acceptance_prob = 1 - ((w_fact % R) / w_fact)
    base_cost = np.polyval(coeffs, w)

    return base_cost / acceptance_prob


def validate_model(coeffs, test_R_values, n_generations=100_000):
    """
    Empirical Validation.
    Compares the theoretical equation against real hardware execution.
    """
    print("\n--- Validating Model ---", file=sys.stderr)

    for R in test_R_values:
        print(f"\nTesting Range R = {R}", file=sys.stderr)
        valid_w = [w for w in range(2, 19) if math.factorial(w) >= R]

        actual_times = []
        theoretical_times = []

        for w in valid_w:
            theoretical = expected_time(w, R, coeffs) * n_generations
            theoretical_times.append(theoretical)

            time.sleep(0.5)  # Cooldown

            start = time.perf_counter_ns()
            generalized_xor_lh(12345, w, R, n_generations)
            end = time.perf_counter_ns()

            actual = (end - start) / 1e9
            actual_times.append(actual)
            print(f"  w={w}: Expected={theoretical:.4f}s, Actual={actual:.4f}s", file=sys.stderr)

        # plotting the validation
        plt.figure(figsize=(8, 5))
        plt.plot(valid_w, actual_times, label='Actual Time', marker='o')
        plt.plot(valid_w, theoretical_times, label='Theoretical Time', linestyle='--')
        plt.title(f"Empirical Validation for Range [0, {R - 1}]")
        plt.xlabel("Window Size (w)")
        plt.ylabel("Time for 100,000 generations (s)")
        plt.legend()
        plt.grid(True)
        plt.show()


def find_optimal_w(R, coeffs):
    """
    Programmatic rule to derive the absolute fastest w.
    """
    best_w = None
    min_time = float('inf')

    for w in range(2, 19):
        if math.factorial(w) >= R:
            t = expected_time(w, R, coeffs)
            if t < min_time:
                min_time = t
                best_w = w

    return best_w


if __name__ == "__main__":
    # get hardware-specific constants
    coeffs, w_vals, t_vals = profile_base_cost()

    # test specific ranges (e.g., a card deck, a calendar, a large sim)
    test_ranges = [52, 365, 1_000_000, 2**32]
    validate_model(coeffs, test_ranges)

    # example usage of the rule:
    example_R = 7200
    w_opt = find_optimal_w(example_R, coeffs)
    print(f"\n--- Derivation ---", file=sys.stderr)
    print(f"For Range {example_R}, the mathematically optimal window size is: w={w_opt}", file=sys.stderr)