import cv2
import numpy as np
import math


def generate_lehmer_sequence(data_array, w):
    """
    Applies the Lehmer code mathematical logic from your paper to an external array.
    This replaces the Xorshift generation step with sliding window data ingestion.
    """
    codes = []
    # Precompute factorials just like in your Cython script
    factorials = [math.factorial(w - i - 1) for i in range(w)]

    # Slide the window of size w across the array (delta = 1)
    for i in range(len(data_array) - w + 1):
        window = data_array[i:i + w]
        lehmer = 0

        # Calculate how many elements to the right are smaller
        for j in range(w):
            smaller = 0
            for k in range(j + 1, w):
                if window[k] < window[j]:
                    smaller += 1
            lehmer += smaller * factorials[j]

        codes.append(lehmer)

    return codes


def extract_ngrams(sequence, n):
    """
    Converts a sequence of Lehmer codes into a set of overlapping n-grams.
    """
    return set(tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1))


def get_multiscale_features(image_path, base_width=250, scales=[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65], w=5,
                            ngram_size=3):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Could not load {image_path}")

    # resize to a normalized base width while maintaining the original aspect ratio
    aspect_ratio = img.shape[0] / img.shape[1]
    base_height = int(base_width * aspect_ratio)
    img_base = cv2.resize(img, (base_width, base_height))

    all_features = set()

    # extract features at multiple scales to create "Scale Invariance"
    for scale in scales:
        scaled_width = int(base_width * scale)
        scaled_height = int(base_height * scale)

        # avoid shrinking too far
        if scaled_width < w or scaled_height < w:
            continue

        img_scaled = cv2.resize(img_base, (scaled_width, scaled_height))
        # blur for digital sensor noise and focus on macro-shapes
        img_scaled = cv2.GaussianBlur(img_scaled, (3, 3), 0)

        # rows
        for row in img_scaled:
            seq = generate_lehmer_sequence(row, w)
            ngrams = extract_ngrams(seq, ngram_size)
            # prefix with 'R'
            all_features.update([("R",) + ng for ng in ngrams])

        # columns
        for col in img_scaled.T:
            seq = generate_lehmer_sequence(col, w)
            ngrams = extract_ngrams(seq, ngram_size)
            # prefix with 'C'
            all_features.update([("C",) + ng for ng in ngrams])

    return all_features


def compare_images_multiscale(path_a, path_b, w=5, ngram_size=3):
    print(f"Comparing:\n1. {path_a}\n2. {path_b}\n")

    # Extract massive feature sets across all scales
    features_a = get_multiscale_features(path_a, base_width=250, w=w, ngram_size=ngram_size)
    features_b = get_multiscale_features(path_b, base_width=250, w=w, ngram_size=ngram_size)

    intersection = len(features_a.intersection(features_b))

    # Jaccard
    union = len(features_a.union(features_b))
    jaccard_score = intersection / union if union > 0 else 0.0

    # Overlap coefficient
    min_features = min(len(features_a), len(features_b))
    containment_score = intersection / min_features if min_features > 0 else 0.0

    print(f"--- Results ---")
    print(f"Total Unique Features A: {len(features_a)}")
    print(f"Total Unique Features B: {len(features_b)}")
    print(f"Shared Features: {intersection}")
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
