"""
############ VAE ###################
python src/scripts/sample_from_model.py \
    --model_type vae \
    --model_name vae \
    --config_path_2d src/results/vae_baseline_1.0/vae_baseline_1.0_config.json \
    --weights_path_2d src/results/vae_baseline_1.0/vae_baseline_1.0_checkpoint.pt \
    --config_path_3d src/results/vae_3d_baseline_1.0_newLoss/vae_3d_baseline_1.0_newLoss_config.json \
    --weights_path_3d src/results/vae_3d_baseline_1.0_newLoss/vae_3d_baseline_1.0_newLoss_checkpoint.pt \
    --upscale_res_2d 28 \
    --upscale_res_3d 32 \
    --n_samples 3 

############ Latent Diffusion ###################
python src/scripts/sample_from_model.py \
    --model_type latent \
    --model_name latent_diffusion \
    --config_path_2d src/train_results/latent-diffusion/metadata/config.json \
    --weights_path_2d src/train_results/latent-diffusion/weights/weights.pt \
    --config_path_3d src/train_results/latent-diffusion-VOXEL-newLoss/metadata/config.json \
    --weights_path_3d src/train_results/latent-diffusion-VOXEL-newLoss/weights/weights.pt \
    --upscale_res_2d 28 \
    --upscale_res_3d 32 \
    --n_samples 3 

    
############ Latent Fixed ###################
python src/scripts/sample_from_model.py \
    --model_type latent \
    --model_name latent_fixed \
    --config_path_2d src/train_results/Latent-two_stage_fixed/Latent-two_stage_fixed_ldm_config.json \
    --weights_path_2d src/train_results/Latent-two_stage_fixed/Latent-two_stage_fixed_ldm_checkpoint.pt \
    --config_path_3d src/train_results/VOXEL-Latent-Fixed-TEST/VOXEL-Latent-Fixed-TEST_ldm_config.json \
    --weights_path_3d src/train_results/VOXEL-Latent-Fixed-TEST/VOXEL-Latent-Fixed-TEST_ldm_checkpoint.pt \
    --upscale_res_2d 28 \
    --upscale_res_3d 32 \
    --n_samples 3 

    
############ Latent Converged ###################
python src/scripts/sample_from_model.py \
    --model_type latent \
    --model_name latent_converge \
    --config_path_2d src/train_results/Latent-two_stage_convergence/Latent-two_stage_convergence_ldm_config.json \
    --weights_path_2d src/train_results/Latent-two_stage_convergence/Latent-two_stage_convergence_ldm_checkpoint.pt \
    --config_path_3d src/train_results/VOXEL-Latent-Converge-TEST/VOXEL-Latent-Converge-TEST_ldm_config.json \
    --weights_path_3d src/train_results/VOXEL-Latent-Converge-TEST/VOXEL-Latent-Converge-TEST_ldm_checkpoint.pt \
    --upscale_res_2d 28 \
    --upscale_res_3d 32 \
    --n_samples 3

############ Weight Diffusion ###################
python src/scripts/sample_from_model.py \
    --model_type weight \
    --model_name weight_diffusion \
    --config_path_2d src/train_results/weight-diffusion/metadata/config.json \
    --weights_path_2d src/train_results/weight-diffusion/weights/weights.pt \
    --config_path_3d src/train_results/VOXEL-Weight-Diffusion-TEST/metadata/config.json \
    --weights_path_3d src/train_results/VOXEL-Weight-Diffusion-TEST/weights/weights.pt \
    --upscale_res_2d 28 \
    --upscale_res_3d 32 \
    --n_samples 3 

    
############ Weight Fixed ###################
python src/scripts/sample_from_model.py \
    --model_type weight \
    --model_name weight_fixed \
    --config_path_2d src/train_results/weight-two-stage-convergence/weight-two-stage-convergence_wd_config.json \
    --weights_path_2d src/train_results/weight-two-stage-convergence/weight-two-stage-convergence_wd_weights.pt \
    --config_path_3d src/train_results/VOXEL-Weight-Converge-TEST/VOXEL-Weight-Converge-TEST_wd_config.json\
    --weights_path_3d src/train_results/VOXEL-Weight-Converge-TEST/VOXEL-Weight-Converge-TEST_wd_weights.pt \
    --upscale_res_2d 28 \
    --upscale_res_3d 32 \
    --n_samples 3 

    
############ Weight Converged ###################
python src/scripts/sample_from_model.py \
    --model_type weight \
    --model_name weight_converged \
    --config_path_2d src/train_results/weight-two-stage-convergence/weight-two-stage-convergence_wd_config.json \
    --weights_path_2d src/train_results/weight-two-stage-convergence/weight-two-stage-convergence_wd_weights.pt \
    --config_path_3d src/train_results/VOXEL-Weight-Converge-TEST/VOXEL-Weight-Converge-TEST_wd_config.json\
    --weights_path_3d src/train_results/VOXEL-Weight-Converge-TEST/VOXEL-Weight-Converge-TEST_wd_weights.pt \
    --upscale_res_2d 28 \
    --upscale_res_3d 32 \
    --n_samples 3    

"""

################ Imports ####################
import argparse
import os
import sys


sys.path.append(".")

from src.utility.general import _get_device
from src.utility.unified_results_eval import (
    prepare_model,
    sample_vectors,
    decode_vectors,
)
from src.utility.plotting import _render_mesh_on_ax, _samples_to_voxel_grids
from src.scripts.get_all_plot_results import make_coord_grid
import matplotlib.pyplot as plt
import numpy as np
import torch


_AZIM_OFFSETS = [120, 120, 120, 120, 120, 120, 120, 120, 120, 120]
_ELEV = 25

# Scale multipliers and their display labels
SCALE_FACTORS = [0.125, 0.25, 0.5, 1, 2, 4]
SCALE_LABELS = ["0.125x", "0.25x", "0.5x", "1x", "2x", "4x"]
###############################################


def _get_arg_parser():
    parser = argparse.ArgumentParser(
        description="Reworked Evaluation Visualizations for INR models."
    )
    parser.add_argument("--config_path_2d", type=str, required=True)
    parser.add_argument("--weights_path_2d", type=str, required=True)
    parser.add_argument("--config_path_3d", type=str, required=True)
    parser.add_argument("--weights_path_3d", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    # Scale input
    parser.add_argument(
        "--upscale_res_2d",
        type=int,
        default=128,
        help="Target resolution for upscaled samples and reconstructions.",
    )
    parser.add_argument(
        "--upscale_res_3d",
        type=int,
        default=32,
        help="Target resolution for upscaled samples and reconstructions.",
    )
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["vae", "latent", "weight"]
    )
    return parser.parse_args()


def _get_model_bundles(args, device):
    print("--- Loading 2D model ---")
    bundle_2d = prepare_model(
        args.model_type,
        args.config_path_2d,
        args.weights_path_2d,
        is_3d=False,
        device=device,
    )
    print("--- Loading 3D model ---")
    bundle_3d = prepare_model(
        args.model_type,
        args.config_path_3d,
        args.weights_path_3d,
        is_3d=True,
        device=device,
    )

    bundle_2d["is_3d"] = False
    bundle_3d["is_3d"] = True

    return bundle_2d, bundle_3d


def sample_and_decode(args, bundle, device, upscale_res, scales):
    print(f"  Sampling {args.n_samples} generated vectors ...")
    gen_vecs, latent_shape = sample_vectors(
        bundle, args.model_type, args.n_samples, args.n_samples, device
    )
    outputs = []
    for scale in scales:
        base_coord_grid = make_coord_grid(
            (int(upscale_res * scale),) * (3 if bundle["is_3d"] else 2),
            (-1, 1),
            device=device,
        )

        print(f"  Decoding {args.n_samples} samples ...")
        gen_decoded = decode_vectors(
            bundle,
            args.model_type,
            gen_vecs,
            latent_shape,
            base_coord_grid,
            args.n_samples,
            device,
        )

        outputs.append(
            convert_to_numoy_image(gen_decoded, bundle["channels"], bundle["is_3d"])
        )

    return outputs


def convert_to_numoy_image(samples, channels, is_3d):
    if not is_3d:
        return _to_numpy_images(samples, channels)
    else:
        return samples


def _to_numpy_images(x_hat: torch.Tensor, channels: int) -> np.ndarray:
    """
    Convert a (B, C, H, W) float tensor in [0,1] to a numpy image array.

    Args:
        x_hat:    (B, C, H, W) float tensor already clamped to [0,1].
        channels: Number of image channels.
    Returns:
        images: (B, H, W) for grayscale or (B, H, W, C) for RGB.
    """
    x_hat = x_hat.cpu().float()
    if channels == 1:
        return x_hat.squeeze(1).numpy()
    return x_hat.permute(0, 2, 3, 1).numpy()


def plot_2d_upscale_sample_row(args, outputs, channels, output_dir, scales):
    i = 0
    for samples in outputs:
        scale = int(args.upscale_res_2d * scales[i])
        fig, axes = plt.subplots(1, args.n_samples, figsize=(args.n_samples * 1.5, 1.5))
        for ax, img in zip(axes, samples, strict=False):
            if channels == 1:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(img, vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")

        plt.subplots_adjust(hspace=0.02, wspace=0.02)
        save_path = os.path.join(output_dir, f"samples_upscaled_{scale}x{scale}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Upscaled samples saved → {save_path}")
        i += 1


def plot_3d_upscale_sample_row(args, output, channels, output_dir, scales):
    i = 0

    for samples in output:
        scale = int(args.upscale_res_3d * scales[i])
        N = args.n_samples
        save_path = os.path.join(
            output_dir, f"samples_3d_upscaled_{scale}x{scale}x{scale}.png"
        )

        recon_grids = _samples_to_voxel_grids(samples, channels, scale)
        recon_grids = recon_grids.transpose(0, 3, 1, 2)  # (N, W, H, D)

        fig = plt.figure(figsize=(N * 2.0, 5.0))
        for col in range(N):
            azim = _AZIM_OFFSETS[col % len(_AZIM_OFFSETS)]

            ax_recon = fig.add_subplot(2, N, N + col + 1, projection="3d")
            _render_mesh_on_ax(ax_recon, recon_grids[col], azim=azim, elev=_ELEV)

        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Reconstruction plot saved -> {save_path}")
        i += 1


def main():
    args = _get_arg_parser()
    device = _get_device()

    output_dir = os.path.join("analysis_results", args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    scales_mnist = [0.25, 0.5, 1, 2, 4, 8]
    scales_shape = [0.25, 0.5, 1, 2, 4]

    print("Loading Models: ")
    bundle_2d, bundle_3d = _get_model_bundles(args, device)

    print("Samples 2d: ")
    outputs = sample_and_decode(
        args, bundle_2d, device, args.upscale_res_2d, scales_mnist
    )
    print("Plots 2d: ")
    plot_2d_upscale_sample_row(
        args, outputs, bundle_2d["channels"], output_dir, scales_mnist
    )

    print("Samples 3d: ")
    outputs = sample_and_decode(
        args, bundle_3d, device, args.upscale_res_3d, scales_shape
    )
    print("Plots 3d: ")
    plot_3d_upscale_sample_row(
        args, outputs, bundle_3d["channels"], output_dir, scales_shape
    )


if __name__ == "__main__":
    main()
