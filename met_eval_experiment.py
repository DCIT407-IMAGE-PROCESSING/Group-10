import os
import pandas as pd
from preprocessing import load_image
from noise import gaussian_noise, salt_pepper_noise, speckle_noise
from filters import mean_filter, gaussian_filter
# ,median_filter, bilateral_filter
from metrics import compute_mse, compute_psnr, compute_ssim

IMAGE_FOLDER = "dataset"
NUM_TRIALS = 3

noise_functions = {
  "Gaussian": gaussian_noise,
  "SaltPepper": salt_pepper_noise,
  "Speckle": speckle_noise
}

filter_functions = {
  "Mean": mean_filter,
  "Gaussian": gaussian_filter,
  # "Median": median_filter,
  # "Bilateral": bilateral_filter
}

def run_experiment():
  results = []

  for image_name in os.listdir(IMAGE_FOLDER):
    path = os.path.join(IMAGE_FOLDER, image_name)
    original = load_image(path)

    for noise_name, noise_func in noise_functions.items():
      for filter_name, filter_func in filter_functions.items():

        mse_list = []
        psnr_list = []
        ssim_list = []

        for trial in range(NUM_TRIALS):
          noisy = noise_func(original)
          filtered = filter_func(noisy)

          mse_list.append(compute_mse(original, filtered))
          psnr_list.append(compute_psnr(original, filtered))
          ssim_list.append(compute_ssim(original, filtered))

        results.append({
          "Image": image_name,
          "Noise": noise_name,
          "Filter": filter_name,
          "MSE": sum(mse_list) / NUM_TRIALS,
          "PSNR": sum(psnr_list) / NUM_TRIALS,
          "SSIM": sum(ssim_list) / NUM_TRIALS
      })

  df = pd.DataFrame(results)
  df.to_csv("final_results.csv", index=False)
  return df
