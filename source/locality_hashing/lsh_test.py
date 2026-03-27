from collections import Counter
import cv2
import numpy as np
import math


def generate_lehmer_sequence(data_array, w, variance_threshold=15):
    """
    Applies the Lehmer code mathematical logic from your paper to an external array.
    This replaces the Xorshift generation step with sliding window data ingestion.
    """
    codes = []
    factorials = [math.factorial(w - i - 1) for i in range(w)]

    for i in range(len(data_array) - w + 1):
        window = data_array[i:i + w]

        # ca to int to prevent numpy uint8 underflow
        if int(max(window)) - int(min(window)) < variance_threshold:
            codes.append(-1)  # insert break token to prevent stitching features
            continue

        lehmer = 0
        for j in range(w):
            smaller = 0
            for k in range(j + 1, w):
                if window[k] < window[j]:
                    smaller += 1
            lehmer += smaller * factorials[j]
        codes.append(lehmer)

    return codes


def extract_ngrams(sequence, n):
    ngrams = [] # change to list to preserve frequency counts
    for i in range(len(sequence) - n + 1):
        chunk = tuple(sequence[i:i + n])
        # only keep n-grams that are physically contiguous, discard break tokens
        if -1 not in chunk:
            ngrams.append(chunk) # append instead of add
    return ngrams


def get_multiscale_features(image_path, base_width=250, scales=[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65], w=5,
                            ngram_size=3):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError(f"Error: Could not load {image_path}")

    # resize to a normalized base width while maintaining the original aspect ratio
    aspect_ratio = img.shape[0] / img.shape[1]
    base_height = int(base_width * aspect_ratio)
    img_base = cv2.resize(img, (base_width, base_height))

    all_features = []  # list instead of set
    # define a spatial grid (4x4) to retain spatial context
    grid_resolution = 4

    for scale in scales:
        scaled_width = int(base_width * scale)
        scaled_height = int(base_height * scale)
        if scaled_width < w or scaled_height < w: continue

        img_scaled = cv2.resize(img_base, (scaled_width, scaled_height))
        # blur for digital sensor noise and focus on macro-shapes
        img_scaled = cv2.GaussianBlur(img_scaled, (3, 3), 0)

        # rows
        for y, row in enumerate(img_scaled):
            seq = generate_lehmer_sequence(row, w)
            ngrams = extract_ngrams(seq, ngram_size)
            grid_y = int((y / img_scaled.shape[0]) * grid_resolution)
            # prefix with 'R'
            all_features.extend([("R", grid_y) + ng for ng in ngrams])  # use extend

        # columns
        for x, col in enumerate(img_scaled.T):
            seq = generate_lehmer_sequence(col, w)
            ngrams = extract_ngrams(seq, ngram_size)
            grid_x = int((x / img_scaled.shape[1]) * grid_resolution)
            # prefix with 'C'
            all_features.extend([("C", grid_x) + ng for ng in ngrams])

    # return a counter dictionary mapping {feature: frequency}
    return Counter(all_features)


def compare_images_multiscale(path_a, path_b, w=5, ngram_size=3):
    print(f"Comparing:\n1. {path_a}\n2. {path_b}\n")

    # these are Counter objects
    counts_a = get_multiscale_features(path_a, base_width=250, w=w, ngram_size=ngram_size)
    counts_b = get_multiscale_features(path_b, base_width=250, w=w, ngram_size=ngram_size)

    total_features_a = sum(counts_a.values())
    total_features_b = sum(counts_b.values())

    # multiset intersection (sum of the minimum frequencies)
    shared_keys = counts_a.keys() & counts_b.keys()
    intersection_count = sum(min(counts_a[k], counts_b[k]) for k in shared_keys)

    # multiset union
    union_count = total_features_a + total_features_b - intersection_count
    jaccard_score = intersection_count / union_count if union_count > 0 else 0.0

    # asymmetric containment
    min_total = min(total_features_a, total_features_b)
    containment_score = intersection_count / min_total if min_total > 0 else 0.0

    print(f"--- Results ---")
    print(f"Total Feature Count A: {total_features_a} (Unique: {len(counts_a)})")
    print(f"Total Feature Count B: {total_features_b} (Unique: {len(counts_b)})")
    print(f"Intersecting Features: {intersection_count}")
    print(f"Jaccard Similarity: {jaccard_score * 100:.2f}%")
    print(f"Asymmetric Containment Score: {containment_score * 100:.2f}%\n")

    return containment_score


if __name__ == "__main__":
    print("Comparing Napoleon")
    compare_images_multiscale("nap1.jpg", "nap2.jpg", w=5, ngram_size=3)
    print("Comparing FKA Twigs")
    compare_images_multiscale("fka1.jpg", "fka2.jpg", w=5, ngram_size=3)
    print("Comparing Napoleon with FKA Twigs")
    compare_images_multiscale("nap1.jpg", "fka2.jpg", w=5, ngram_size=3)
