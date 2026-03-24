import os

FID_SCORE_SAMPLES = 10_000
FID_SAMPLE_BATCH = 2000
CLASSIFIER_WEIGHTS = "src/results/classifier/weights.pth"
CLASSIFIER_CONFIG = "src/results/classifier/config.json"
CACHE_DIR = "src/results/cache"
CACHE_PATH = os.path.join(CACHE_DIR, "real_mnist_features.npz")

MODEL_LABELS = {
    "ndm": "NDM",
    "inr_vae": "VAE-INR",
    "ndm_inr": "NDM-INR",
}
MODEL_COLORS = {
    "ndm": "#2a6fdb",
    "inr_vae": "#e07b39",
    "ndm_inr": "#2ca05a",
}

SAMPLE_COMPARISON_GRID_SIZE = 6  # n x n grid per model

# =============================================================================
# Config
# =============================================================================

NUM_UPSCALING_IMAGES = 3
UPSCALED_RESOLUTIONS = [28, 64, 128, 256, 512, 1024]
