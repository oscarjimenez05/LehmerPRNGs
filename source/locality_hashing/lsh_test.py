import cv2
import numpy as np
import math


def get_row_averages(image_path, target_height=500):
    """
    Loads an image, converts it to grayscale, resizes it to a uniform
    vertical height, and calculates the average pixel intensity per row.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Could not load image at {image_path}. Check the path.")

    # Maintain aspect ratio while normalizing the vertical height
    aspect_ratio = img.shape[1] / img.shape[0]
    target_width = int(target_height * aspect_ratio)
    img_resized = cv2.resize(img, (target_width, target_height))

    # Compress 2D matrix to 1D array of row averages
    row_averages = np.mean(img_resized, axis=1)
    return row_averages


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


def calculate_jaccard_similarity(set_a, set_b):
    """
    Calculates the Jaccard index between two sets.
    """
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return intersection / union


def compare_images(path_a, path_b, w=5, ngram_size=3):
    """
    Orchestrates the pipeline to compare two images using Lehmer Code LSH.
    """
    print(f"Comparing:\n1. {path_a}\n2. {path_b}\n")

    arr_a = get_row_averages(path_a)
    arr_b = get_row_averages(path_b)

    seq_a = generate_lehmer_sequence(arr_a, w)
    seq_b = generate_lehmer_sequence(arr_b, w)

    # Shingles (n-grams)
    shingles_a = extract_ngrams(seq_a, ngram_size)
    shingles_b = extract_ngrams(seq_b, ngram_size)

    # Calculate Similarity
    score = calculate_jaccard_similarity(shingles_a, shingles_b)

    print(f"--- Results ---")
    print(f"Window Size (w): {w}")
    print(f"N-Gram Size: {ngram_size}")
    print(f"Unique Lehmer n-grams in Image A: {len(shingles_a)}")
    print(f"Unique Lehmer n-grams in Image B: {len(shingles_b)}")
    print(f"Shared n-grams: {len(shingles_a.intersection(shingles_b))}")
    print(f"Similarity Score: {score * 100:.2f}%\n")

    return score


if __name__ == "__main__":
    # paths for the example execution
    image_1 = "img1.jpg"
    image_2 = "img2.jpg"

    try:
        compare_images(image_1, image_2, w=6, ngram_size=4)


    except Exception as e:
        print(e)