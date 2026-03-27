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


def get_2d_lehmer_features(image_path, grid_size=64, w=5, ngram_size=3):
    """
    Extracts a 2D Bag-of-Words Lehmer fingerprint from an image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Could not load image at {image_path}.")

    # force into fixed grid and blur slightly (low pass filter)
    img_resized = cv2.resize(img, (grid_size, grid_size))
    img_blurred = cv2.GaussianBlur(img_resized, (3, 3), 0)

    all_ngrams = set()

    # extract features from every row
    for row in img_blurred:
        seq = generate_lehmer_sequence(row, w)
        ngrams = extract_ngrams(seq, ngram_size)
        # Prefix 'R' to designate horizontal features
        all_ngrams.update([("R",) + ng for ng in ngrams])

    # extract features from every column
    for col in img_blurred.T:
        seq = generate_lehmer_sequence(col, w)
        ngrams = extract_ngrams(seq, ngram_size)
        # Prefix 'C' to designate vertical features
        all_ngrams.update([("C",) + ng for ng in ngrams])

    return all_ngrams


def compare_images_2d(path_a, path_b, w=5, ngram_size=3):
    print(f"Comparing:\n1. {path_a}\n2. {path_b}\n")

    features_a = get_2d_lehmer_features(path_a, grid_size=64, w=w, ngram_size=ngram_size)
    features_b = get_2d_lehmer_features(path_b, grid_size=64, w=w, ngram_size=ngram_size)

    # Calculate Jaccard Similarity
    intersection = len(features_a.intersection(features_b))
    union = len(features_a.union(features_b))
    score = intersection / union if union > 0 else 0.0

    print(f"--- Results ---")
    print(f"Window Size (w): {w}")
    print(f"N-Gram Size: {ngram_size}")
    print(f"Total Unique Features A: {len(features_a)}")
    print(f"Total Unique Features B: {len(features_b)}")
    print(f"Shared Features: {intersection}")
    print(f"Similarity Score: {score * 100:.2f}%\n")

    return score


if __name__ == "__main__":
    compare_images_2d("img1.jpg", "img2.jpg", w=5, ngram_size=3)