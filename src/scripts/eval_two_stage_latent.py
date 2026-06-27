"""

python src/scripts/eval_two_stage_latent.py \
    --run_name latent_two_stage_fixed \
    --run_dir src/train_results/latent_two_stage_fixed \
    --ldm_config src/train_results/latent_two_stage_fixed/latent_two_stage_fixed_ldm_config.json \
    --n_fid_samples 128 \
    --fid_batch_size 128


python src/scripts/eval_two_stage_latent.py \
    --run_name two_stage_convergence \
    --run_dir src/train_results/two_stage_convergence \
    --ldm_config src/train_results/two_stage_convergence/two_stage_convergence_ldm_config.json \
    --n_fid_samples 128 \
    --fid_batch_size 128


"""

import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(".")

from src.utility.dataset_builders import build_dataset
from src.utility.model_builders.util.vae_builder import load_pretrained_vae
from src.utility.model_builders.util.twostage_builder import build_ldm
from src.utility.evaluation import compute_final_eval


# ----------------------------
# ARGS
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--ldm_config", type=str, required=True)
    p.add_argument("--n_fid_samples", type=int, default=1024)
    p.add_argument("--fid_batch_size", type=int, default=64)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--mode", type=str, default="eval")

    return p.parse_args()


# ----------------------------
# MAIN
# ----------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.ldm_config, "r") as f:
        hparams = json.load(f)

    dataset, val_dataset, data_config = build_dataset(
        dataset_name=hparams["dataset"],
        data_root="data/",
        subset_frac=1.0,
        single_class=False,
    )

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    results_dir = args.run_dir

    # ----------------------------
    # load VAE
    # ----------------------------
    vae_path = os.path.join(
        results_dir,
        f"{hparams['run_name']}_vae_weights.pt"
    )

    vae = load_pretrained_vae(
        vae_path,
        hparams,
        data_config["channels"],
        data_config["img_size"],
        device,
        is_3d=data_config.get("is_3d", False),
    )

    # ----------------------------
    # load LDM checkpoint
    # ----------------------------
    ldm_ckpt = torch.load(
        os.path.join(results_dir, f"{hparams['run_name']}_ldm_weights.pt"),
        map_location=device,
    )

    # minimal args required by build_ldm
    from types import SimpleNamespace

    ldm_args = SimpleNamespace(
        T=hparams["T"],
        beta_1=hparams["beta_1"],
        beta_T=hparams["beta_T"],
        lr=1e-4,
        weight_decay=0.0,
        grad_clip=0.0,
        ddpm_max_epochs=1,
        ddpm_patience=1,
        ddpm_delta=0.0,
        fid_fractions=[1.0],
        ddpm_check_every=1,
    )

    ldm = build_ldm(
        hparams=hparams,
        args=ldm_args,
        channels=data_config["channels"],
        img_size=data_config["img_size"],
        device=device,
        is_3d=data_config.get("is_3d", False),
    )

    # IMPORTANT: matches your training save format
    ldm.load_state_dict(ldm_ckpt["ldm_state_dict"], strict=True)
    ldm.eval()

    # ----------------------------
    # FINAL EVAL (single source of truth)
    # ----------------------------
    compute_final_eval(
        vae=vae,
        ldm=ldm,
        hparams=hparams,
        val_loader=val_loader,
        data_config=data_config,
        args=args,
        results_dir=results_dir,
        device=device,
        vae_epochs=0,          # if you are loading pretrained VAE
        ddpm_epochs=hparams.get("ddpm_epochs", 0),
        skip_vae_eval=False,
    )


if __name__ == "__main__":
    main()