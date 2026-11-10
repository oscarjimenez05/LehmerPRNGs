import os
import glob
import cv2
import numpy as np
import image_transforms as T
import minhash_lsh as ML
from lehmer_lsh_test import get_multiscale_features


def compute_lehmer_similarity(counts_a, counts_b):
    shared = counts_a.keys() & counts_b.keys()
    intersection = sum(min(counts_a[k], counts_b[k]) for k in shared)
    min_total = min(sum(counts_a.values()), sum(counts_b.values()))
    return intersection / min_total if min_total > 0 else 0.0


def run_benchmark(image_folder="./eval_images"):
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.[jJ][pP][gG]")) +
                         glob.glob(os.path.join(image_folder, "*.[pP][nN][gG]")))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_folder}!")

    results = {
        "Lehmer": {
            "Monotonic": [],
            "Crop": [],
            "NoiseBlur": [],
            "Occlusion": [],
            "Distinct": [],
        },
        "MinHash": {
            "Monotonic": [],
            "Crop": [],
            "NoiseBlur": [],
            "Occlusion": [],
            "Distinct": [],
        },
    }

    print(f"Running benchmark on {len(image_paths)} diverse images...")

    for i, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 1. Synthesize ground-truth transformations
        mono = T.apply_monotonic_distortion(img)
        crop = T.apply_crop(img)
        blur = T.apply_noise_blur(img)
        occl = T.apply_occlusion(img)

        # We save temp versions for lsh_test.get_multiscale_features which expects a path
        cv2.imwrite("temp_orig.jpg", img)
        cv2.imwrite("temp_mono.jpg", mono)
        cv2.imwrite("temp_crop.jpg", crop)
        cv2.imwrite("temp_blur.jpg", blur)
        cv2.imwrite("temp_occl.jpg", occl)

        f_orig = get_multiscale_features("temp_orig.jpg")
        f_mono = get_multiscale_features("temp_mono.jpg")
        f_crop = get_multiscale_features("temp_crop.jpg")
        f_blur = get_multiscale_features("temp_blur.jpg")
        f_occl = get_multiscale_features("temp_occl.jpg")

        # 2. MinHash signatures directly on numpy arrays
        m_orig = ML.get_minhash_signature(img)
        m_mono = ML.get_minhash_signature(mono)
        m_crop = ML.get_minhash_signature(crop)
        m_blur = ML.get_minhash_signature(blur)
        m_occl = ML.get_minhash_signature(occl)

        # True Positive scores
        results["Lehmer"]["Monotonic"].append(compute_lehmer_similarity(f_orig, f_mono))
        results["Lehmer"]["Crop"].append(compute_lehmer_similarity(f_orig, f_crop))
        results["Lehmer"]["NoiseBlur"].append(compute_lehmer_similarity(f_orig, f_blur))
        results["Lehmer"]["Occlusion"].append(compute_lehmer_similarity(f_orig, f_occl))

        results["MinHash"]["Monotonic"].append(ML.compute_minhash_similarity(m_orig, m_mono))
        results["MinHash"]["Crop"].append(ML.compute_minhash_similarity(m_orig, m_crop))
        results["MinHash"]["NoiseBlur"].append(ML.compute_minhash_similarity(m_orig, m_blur))
        results["MinHash"]["Occlusion"].append(ML.compute_minhash_similarity(m_orig, m_occl))

        # Negative baseline: compare against the previous distinct image in the set
        if i > 0:
            prev_img = cv2.imread(image_paths[i - 1], cv2.IMREAD_GRAYSCALE)
            cv2.imwrite("temp_prev.jpg", prev_img)
            f_prev = get_multiscale_features("temp_prev.jpg")
            m_prev = ML.get_minhash_signature(prev_img)

            results["Lehmer"]["Distinct"].append(compute_lehmer_similarity(f_orig, f_prev))
            results["MinHash"]["Distinct"].append(ML.compute_minhash_similarity(m_orig, m_prev))

    # Clean up temp files
    for t in ["temp_orig.jpg", "temp_mono.jpg", "temp_crop.jpg", "temp_blur.jpg", "temp_prev.jpg"]:
        if os.path.exists(t):
            os.remove(t)

    # Output formatted report
    print("\n" + "=" * 65)
    print(f"{'Condition':<20} | {'Lehmer LSH (Mean)':<20} | {'MinHash (Mean)':<20}")
    print("=" * 65)
    for cond in ["Monotonic", "Crop", "NoiseBlur", "Occlusion", "Distinct"]:
        lehmer_mean = np.mean(results["Lehmer"][cond]) * 100
        minhash_mean = np.mean(results["MinHash"][cond]) * 100
        print(f"{cond:<20} | {lehmer_mean:>18.2f}% | {minhash_mean:>18.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()