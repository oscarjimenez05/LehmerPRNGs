import csv
import random
import glob
import numpy as np
import matplotlib.pyplot as plt


def analyze_and_plot_aggregated(file_pattern="Optimal Window Size*.csv"):
    csv_files = glob.glob(file_pattern)
    if not csv_files:
        print("No CSV files found matching the pattern.")
        return

    print(f"Found {len(csv_files)} files. Aggregating data...")

    all_valid_trials = []

    for filename in csv_files:
        ranges_data = {}
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if not row.get('Tested_w') or str(row['Tested_w']).strip() == '':
                    continue

                R = row['Range_R']
                w = int(row['Tested_w'])

                if R not in ranges_data:
                    ranges_data[R] = {'options': [], 'predicted': None, 'actual': None}

                ranges_data[R]['options'].append(w)
                if int(row['Is_Predicted_Optimal']) == 1:
                    ranges_data[R]['predicted'] = w
                if int(row['Is_Actual_Optimal']) == 1:
                    ranges_data[R]['actual'] = w

        for R, data in ranges_data.items():
            if data['predicted'] is not None and data['actual'] is not None:
                all_valid_trials.append(data)

    print(f"Total independent trials aggregated: {len(all_valid_trials)}")

    model_exact = 0
    model_distance = 0
    for trial in all_valid_trials:
        model_exact += (trial['actual'] == trial['predicted'])
        # model_distance += abs(trial['actual'] - trial['predicted'])
        model_distance_sum = sum(abs(t['actual'] - t['predicted']) for t in all_valid_trials)
        model_mae = model_distance_sum / len(all_valid_trials)

    print("Running 10,000 Monte Carlo simulations...")
    simulations = 10000
    random_exacts = []
    random_distances = []

    for _ in range(simulations):
        sim_e = 0
        sim_d = 0
        for trial in all_valid_trials:
            guess = random.choice(trial['options'])
            sim_e += (guess == trial['actual'])
            sim_d += abs(trial['actual'] - guess)
        random_exacts.append(sim_e)
        # random_distances.append(sim_d)
        random_distances.append(sim_d / len(all_valid_trials))

    # p-values
    p_exact = max(sum(1 for x in random_exacts if x >= model_exact) / simulations, 1 / simulations)
    #p_dist = max(sum(1 for x in random_distances if x <= model_distance) / simulations, 1 / simulations)
    p_dist = max(sum(1 for x in random_distances if x <= model_mae) / simulations, 1 / simulations)

    # ==========================================
    # plots
    plt.style.use('bmh')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Unified Model Performance vs. Random Chance ({len(all_valid_trials)} Aggregated Trials)",
                 fontsize=16, fontweight='bold')

    # --- Exact Matches ---
    ax1.hist(random_exacts, bins=range(max(random_exacts) + 2), align='left', color='#4C72B0', edgecolor='white',
             alpha=0.8)
    ax1.axvline(model_exact, color='#C44E52', linestyle='dashed', linewidth=3, label=f"Theoretical Model ({model_exact})")

    ax1.set_title("Exact Optimal Matches", fontsize=14)
    ax1.set_xlabel("Number of Correct Predictions")
    ax1.set_ylabel("Frequency (out of 10,000 sims)")
    ax1.legend()
    textstr1 = f"Expected by Chance: {np.mean(random_exacts):.1f}\np-value: {p_exact:.4f}"
    ax1.text(0.95, 0.5, textstr1, transform=ax1.transAxes, fontsize=12,
             verticalalignment='center', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # --- Total Distance (Near Misses) ---
    min_d, max_d = min(random_distances), max(random_distances)
    ax2.hist(random_distances, bins=30, color='#55A868', edgecolor='white', alpha=0.8)

    # ax2.axvline(model_distance, color='#C44E52', linestyle='dashed', linewidth=3,
    #             label=f"Theoretical Model (Dist: {model_distance})")
    #
    # ax2.set_title("Total Absolute Distance", fontsize=14)
    # ax2.set_xlabel("Cumulative Distance from Actual Optimal $w$")

    ax2.axvline(model_mae, color='#C44E52', linestyle='dashed', linewidth=3,
                label=f"Theoretical Model (MAE: {model_mae:.2f})")
    ax2.set_title("Mean Absolute Error", fontsize=14)
    ax2.set_xlabel("Average Window Steps from Actual Optimal $w$")
    ax2.set_ylabel("Frequency (out of 10,000 sims)")
    ax2.legend()

    textstr2 = f"Expected Distance: {np.mean(random_distances):.1f}\np-value: {p_dist:.4f}"
    ax2.text(0.05, 0.5, textstr2, transform=ax2.transAxes, fontsize=12,
             verticalalignment='center', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('unified_monte_carlo_results2.png', dpi=300)
    print("Analysis complete. High-res unified plot saved as 'unified_monte_carlo_results.png'")


if __name__ == "__main__":
    analyze_and_plot_aggregated("results/Optimal Window Size*.csv")