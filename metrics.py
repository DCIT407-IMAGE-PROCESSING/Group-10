import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_mse(original, processed):
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)
    return np.mean((original - processed) ** 2)


def compute_psnr(original, processed):
    mse = compute_mse(original, processed)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)


def compute_ssim(original, processed):
    return ssim(original, processed, data_range=255)
