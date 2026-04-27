import time
import math
import csv
import random
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="Validate timing constants.")
parser.add_argument('constants', type=float, nargs=3, help='The three constants')
HARDWARE_COEFFS = list(parser.parse_args().constants)

import xor_lh as xlh
def generalized_xor_lh(seed, w, R, n):
    generator = xlh.XorLehmer(seed, w, 0, 0, R - 1)
    return generator.generate_chunk(n, 0)


def expected_time(w, R, coeffs):
    """Calculates theoretical expected time per generation."""
    w_fact = math.factorial(w)
    if w_fact < R:
        return float('inf')

    acceptance_prob = 1 - ((w_fact % R) / w_fact)
    base_cost = np.polyval(coeffs, w)
    return base_cost / acceptance_prob


def find_predicted_optimal_w(R, coeffs):
    """Finds the theoretical fastest window size for a given range."""
    best_w = None
    min_time = float('inf')
    for w in range(2, 19):
        if math.factorial(w) >= R:
            t = expected_time(w, R, coeffs)
            if t < min_time:
                min_time = t
                best_w = w
    return best_w


def generate_test_ranges():
    """Generates standard, random, and edge-case ranges for rigorous testing."""
    ranges = {10, 52, 365, 1000, 10_000, 1_000_000, 16_777_216}  # Standards (Cards, Days, Hex colors)

    # Edge cases based on factorials
    for w in range(4, 19):
        fact = math.factorial(w)
        ranges.add((fact // 2) + 1)  # around 50% discard rate (Worst Case)
        ranges.add((fact // 3) + 1)  # 33% discard rate
        ranges.add((fact // 4) + 1)  # 25% discard rate
        ranges.add(fact - 1)  # 0% discard rate (Best Case)

    # a few random numbers to fill out the dataset
    for _ in range(15):
        ranges.add(random.randint(100, 5_000_000))

    return sorted(list(ranges))


def run_csv_validation_suite(n_generations=50_000, filename="lehmer_validation_results2.csv"):
    """
    Runs the full battery of tests and outputs a CSV.
    """
    test_ranges = generate_test_ranges()

    print(f"Starting Validation Suite for {len(test_ranges)} unique ranges...")
    print(f"Data will be saved to '{filename}'\n")

    # Open CSV for writing
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow([
            "Range_R",
            "Tested_w",
            "Discard_Percentage",
            "Predicted_Time_Sec",
            "Actual_Time_Sec",
            "Is_Predicted_Optimal",
            "Is_Actual_Optimal"
        ])

        for R in test_ranges:
            print(f"Testing Range: R = {R:,}")
            valid_w_list = [w for w in range(2, 19) if math.factorial(w) >= R]

            # bound the testing to first 6 valid windows
            valid_w_list = valid_w_list[:6]

            predicted_opt_w = find_predicted_optimal_w(R, HARDWARE_COEFFS)

            range_results = []

            for w in valid_w_list:
                # theoretical
                theoretical_t = expected_time(w, R, HARDWARE_COEFFS) * n_generations
                discard_pct = ((math.factorial(w) % R) / math.factorial(w)) * 100

                # prevent thermal throttling
                time.sleep(0.3)

                # actual
                start = time.perf_counter_ns()
                generalized_xor_lh(12345, w, R, n_generations)
                end = time.perf_counter_ns()
                actual_t = (end - start) / 1e9

                # data row
                range_results.append({
                    "w": w,
                    "discard": round(discard_pct, 2),
                    "pred_t": theoretical_t,
                    "act_t": actual_t,
                    "is_pred_opt": int(w == predicted_opt_w)
                })

            # Determine which window was ACTUALLY the fastest in our empirical trial
            fastest_actual_w = min(range_results, key=lambda x: x["act_t"])["w"]

            # Write all rows for this Range to the CSV
            for res in range_results:
                writer.writerow([
                    R,
                    res["w"],
                    res["discard"],
                    f"{res['pred_t']:.6f}",
                    f"{res['act_t']:.6f}",
                    res["is_pred_opt"],
                    int(res["w"] == fastest_actual_w)
                ])

    print(f"\n Validation complete! File saved as {filename}.")


if __name__ == "__main__":
    print(f"Validating timing constants ({HARDWARE_COEFFS[0]}, {HARDWARE_COEFFS[1]}, {HARDWARE_COEFFS[2]})")
    run_csv_validation_suite(n_generations=50_000)