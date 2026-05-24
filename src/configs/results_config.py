import os

FID_SCORE_SAMPLES = 10_000
FID_SAMPLE_BATCH = 1000

CLASSIFIER_WEIGHTS = "src/results/classifier/weights.pth"
CLASSIFIER_CONFIG = "src/results/classifier/config.json"

CACHE_DIR = "src/results/cache"
CACHE_PATH = os.path.join(CACHE_DIR, "real_mnist_features.npz")

MODEL_LABELS = {
    "latent_inr_diffusion": "Latent INR Diffusion",
    "ndm_static_transinr": "NDM Static TransINR",
    "transinr_vae": "TransINR VAE",
}

MODEL_COLORS = {
    "latent_inr_diffusion": "#2a6fdb",
    "ndm_static_transinr": "#e07b39",
    "transinr_vae": "#2ca05a",
}

SAMPLE_COMPARISON_GRID_SIZE = 6

# =============================================================================
# Config
# =============================================================================
NUM_UPSCALING_IMAGES = 3
UPSCALED_RESOLUTIONS = [28, 64, 128, 256, 512, 1024]
