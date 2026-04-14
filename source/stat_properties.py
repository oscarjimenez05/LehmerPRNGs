#!/usr/local/bin/python3
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

from generators import *
import c_lcg_lh
import xor_lh
from alternatives import logistic_lh
from scipy.stats import chisquare
import statsmodels.api as sm


def large_lcg_vs_lcg_lh():
    seed = 135
    reps = 10000
    window = 6

    max_exclusive = math.factorial(window)
    print("Checkpoint 1.1")
    a_lcg = [(seed*421 + 1) % max_exclusive]
    for i in range(reps-1):
        a_lcg.append((a_lcg[i]*421+1) % max_exclusive)
    assert len(a_lcg) == reps
    a_lcg = np.array(a_lcg)

    print("Checkpoint 1.2")
    pre = (seed * 1664525 + 1013904223) % max_exclusive
    a_large_lcg = [pre]
    for i in range(reps-1):
        a_large_lcg.append(pre % max_exclusive)
        pre = (a_large_lcg[i] * 1664525 + 1013904223) % 2**32

    assert len(a_large_lcg) == reps
    a_large_lcg = np.array(a_large_lcg)

    print("Checkpoint 2")
    LCG_LH = c_lcg_lh.LcgLehmer(seed, window, 0, 0, max_exclusive - 1)
    a_lcg_lh64 = LCG_LH.generate_chunk(reps, 0)
    a_lcg_lh64 = np.array(a_lcg_lh64)

    XOR_LH = xor_lh.XorLehmer(seed, window, 0, 0, max_exclusive - 1)
    a_xor_lh64 = XOR_LH.generate_chunk(reps, 0)
    a_xor_lh64 = np.array(a_xor_lh64)

    a_csprng = csprng(reps, max_exclusive)
    a_csprng = np.array(a_csprng)

    data = [("CSPRNG", a_csprng), ("LCG   ", a_lcg), ("Lm_LCG", a_large_lcg), ("LCG_LH", a_lcg_lh64), ("XOR_LH", a_xor_lh64)]
    display_arrays(data, max_exclusive)

    print("Checkpoint 3")
    # Plot histograms
    plot_distribution(a_csprng, "CSPRNG", 2*9*5)
    plot_distribution(a_lcg, "LCG", 2*9*5)
    plot_distribution(a_large_lcg, "Lm_LCG", 2*9*5)
    plot_distribution(a_lcg_lh64, "LCG_LH", 2*9*5)
    plot_distribution(a_xor_lh64, "XOR_LH", 2*9*5)

    print("-----------------------")
    # chisq test
    for title, array in data:
        counts, _ = np.histogram(array, bins=max_exclusive, range=(0, max_exclusive))
        chi2, p = chisquare(counts)
        print(f"{title} Chi^2 statistic = {chi2:.2f}, p-value = {p:.5f}")


def missing_from_range(lst: [int], start: int, end: int) -> [int]:
    """
    :param lst: a list of numbers
    :param start: start value (inclusive)
    :param end: end value (inclusive)
    :return: a list of numbers not present in the range
    """
    full_range = set(range(start, end + 1))
    return sorted(full_range - set(lst))


def serial_correlation_comparison():
    seed = 135
    reps = 100000
    window = 6

    max_exclusive = math.factorial(window)

    a_lcg = [(seed * 421 + 1) % max_exclusive]
    for i in range(reps - 1):
        a_lcg.append((a_lcg[i] * 421 + 1) % max_exclusive)
    assert len(a_lcg) == reps
    a_lcg = np.array(a_lcg)

    pre = (seed * 1664525 + 1013904223) % max_exclusive
    a_large_lcg = [pre]
    for i in range(reps - 1):
        a_large_lcg.append(pre % max_exclusive)
        pre = (a_large_lcg[i] * 1664525 + 1013904223) % 2 ** 32

    assert len(a_large_lcg) == reps
    a_large_lcg = np.array(a_large_lcg)

    LCG_LH = c_lcg_lh.LcgLehmer(seed, window, 0, 0, max_exclusive - 1)
    a_lcg_lh64 = LCG_LH.generate_chunk(reps, 0)

    XOR_LH = xor_lh.XorLehmer(seed, window, 0, 0, max_exclusive - 1)
    a_xor_lh64 = XOR_LH.generate_chunk(reps, 0)

    a_csprng = csprng(reps, max_exclusive)
    a_mrs_tw = mrs_tw(seed, reps, max_exclusive)

    print("-----------------------")
    lags = [1, window, max_exclusive, max_exclusive + 1]
    print("CSPRNG:")
    ljung_box_results = sm.stats.acorr_ljungbox(a_csprng, lags=lags, return_df=True)
    print(ljung_box_results)

    print("LCG:")
    ljung_box_results = sm.stats.acorr_ljungbox(a_lcg, lags=lags, return_df=True)
    print(ljung_box_results)

    print("Lm_LCG:")
    ljung_box_results = sm.stats.acorr_ljungbox(a_large_lcg, lags=lags, return_df=True)
    print(ljung_box_results)

    print("LCG_LH:")
    ljung_box_results = sm.stats.acorr_ljungbox(a_lcg_lh64, lags=lags, return_df=True)
    print(ljung_box_results)

    print("XOR_LH:")
    ljung_box_results = sm.stats.acorr_ljungbox(a_xor_lh64, lags=lags, return_df=True)
    print(ljung_box_results)

    print("MRS_TW:")
    ljung_box_results = sm.stats.acorr_ljungbox(a_mrs_tw, lags=lags, return_df=True)
    print(ljung_box_results)


def display_arrays(data: [Tuple[str, list]], max_exclusive: int, plot: bool = False) -> None:
    """
    :param data: a list of tuples of the form (title, array)
    :param max_exclusive: range of numbers generated, 0 to max_exclusive-1
    :param plot: whether to plot the data or not, defaults to False
    :return: None
    """
    general_display_arrays(data, 0, max_exclusive-1, plot)


def general_display_arrays(data: [Tuple[str, list]], minimum: int, maximum: int, plot: bool = False) -> None:
    """
    :param data: a list of tuples of the form (title, array)
    :param minimum: (inclusive)
    :param maximum: (inclusive)
    :param plot: whether to plot the data or not, defaults to False
    :return: None
    """
    if plot:
        for title, array in data:
            plt.plot(array)
            plt.title(title)
            plt.xlabel("Index of value generated")
            plt.ylabel("Value Generated")
            plt.savefig("indexvalue.png")

    print("-----------------------")

    for title, array in data:
        print(f"{title} MIN and MAX: " + str(int(array.min())) + ", " + str(int(array.max())))
    print("-----------------------")
    for title, array in data:
        print(f"Not present in {title}: " + str(missing_from_range(array, minimum, maximum)))
    print("-----------------------")
    exp_mean = (maximum + minimum) / 2
    print("Expec. MEAN: " + str(exp_mean))
    for title, array in data:
        mean = array.mean()
        print(f"{title} MEAN: " + str(mean) + f" (diff. {(exp_mean - mean):.5f})")


def plot_distribution(data, title="Distribution of Values", bins=24):
    plt.hist(data, bins=bins, align="left", rwidth=0.9, color="skyblue", edgecolor="black")
    plt.xlabel("Generated Values")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig("distribution.png")


if __name__ == "__main__":
    print('Checkpoint 1')
    large_lcg_vs_lcg_lh()
    print("Checkpoint 4")
    # generator = logistic_lh.LogisticLehmer(123456789, 14, 0, 0, 5039)
    # array = generator.generate_chunk(1000000, 0)
    # general_display_arrays([("log_lh", array)], 0, 5039)
    # plot_distribution(array, bins=5040)
    serial_correlation_comparison()
