import cv2
import numpy as np


def apply_monotonic_distortion(img_gray):
  """Random non-linear tone curve: strictly preserves ordinal pixel rankings."""
  # Random gamma between 0.5 (brightening) and 2.2 (darkening)
  gamma = np.random.uniform(0.5, 2.2)
  # Random contrast multiplier
  scale = np.random.uniform(0.8, 1.4)

  arr = img_gray.astype(np.float32) / 255.0
  arr = np.clip(np.power(arr, gamma) * scale, 0.0, 1.0)
  return (arr * 255.0).astype(np.uint8)


def apply_crop(img_gray, min_fraction=0.75, max_fraction=0.90):
  """Simulates variable framing shifts with randomized crop offsets."""
  h, w = img_gray.shape
  fraction = np.random.uniform(min_fraction, max_fraction)

  new_h, new_w = int(h * fraction), int(w * fraction)
  # Random top-left offset
  start_y = np.random.randint(0, h - new_h + 1)
  start_x = np.random.randint(0, w - new_w + 1)

  return img_gray[start_y : start_y + new_h, start_x : start_x + new_w]


def apply_noise_blur(img_gray):
  """Random Gaussian blur radius and randomized additive Gaussian noise."""
  # Random kernel size: 3, 5, or 7
  ksize = np.random.choice([3, 5, 7])
  sigma = np.random.uniform(0.8, 2.0)
  blurred = cv2.GaussianBlur(img_gray, (ksize, ksize), sigma)

  # Randomized noise variance
  noise_sigma = np.random.uniform(5.0, 20.0)
  noise = np.random.normal(0, noise_sigma, blurred.shape).astype(np.int16)

  return np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_occlusion(img_gray, min_size=0.15, max_size=0.30):
  """Mimics real-world obstructions (e.g.
  the priest's head in Case 1) by injecting a random textured patch into a
  quadrant.
  """
  h, w = img_gray.shape
  occ_h = int(h * np.random.uniform(min_size, max_size))
  occ_w = int(w * np.random.uniform(min_size, max_size))

  # Random location (often lower quadrants for bystanders/foreground)
  start_y = np.random.randint(0, h - occ_h + 1)
  start_x = np.random.randint(0, w - occ_w + 1)

  res = img_gray.copy()

  # Create an occlusion: either a noisy textured patch or a random solid block
  if np.random.rand() > 0.5:
    patch = np.random.randint(0, 256, (occ_h, occ_w), dtype=np.uint8)
    patch = cv2.GaussianBlur(patch, (5, 5), 0)
  else:
    patch = np.full(
        (occ_h, occ_w), np.random.randint(20, 220), dtype=np.uint8
    )

  res[start_y : start_y + occ_h, start_x : start_x + occ_w] = patch
  return res