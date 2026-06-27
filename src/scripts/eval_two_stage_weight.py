"""

python src/scripts/eval_two_stage_weight.py \
    --run_dir src/train_results/wd_two_stage_fixed \
    --wd_config src/train_results/wd_two_stage_fixed/wd_two_stage_fixed_wd_config.json \
    --n_fid_samples 128 \
    --fid_batch_size 128


python src/scripts/eval_two_stage_weight.py \
    --run_dir src/train_results/wd_two_stage_convergence \
    --wd_config src/train_results/wd_two_stage_convergence/wd_two_stage_convergence_wd_config.json \
    --n_fid_samples 128 \
    --fid_batch_size 128

"""


import argparse
import json
import os
import torch
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.append(".")
from src.utility.dataset_builders import build_dataset
from src.utility.classifier_utils import (
    _get_inception,
    _inception_features,
    _load_classifier,
    _load_or_compute_real_features,
)
from src.utility.metrics_util import _fid
from src.models.weight_diffusion.WeightDiffusion import WeightDiffusion
from src.scripts.two_stage_weight_training import build_full_wd_model

# ----------------------------
# ARGUMENTS
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--wd_config", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_fid_samples", type=int, default=1024)
    p.add_argument("--fid_batch_size", type=int, default=64)
    return p.parse_args()


# ----------------------------
# LOAD MODEL
# ----------------------------

def load_model(run_dir: str, config: dict, device: torch.device, data_config: dict):
    ckpt_path = os.path.join(
        run_dir,
        f"{config['run_name']}_wd_weights.pt"
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_full_wd_model(
        hparams=config,
        args=argparse.Namespace(
            T=config["T"],
            beta_1=config["beta_1"],
            beta_T=config["beta_T"],
        ),
        channels=data_config["channels"],
        img_size=data_config["img_size"],
        data_dim=data_config["data_dim"],
        device=device,
    )

    state_dict = ckpt["full_model_state_dict"]

    if "coords" in state_dict:
        del state_dict["coords"]

    model.load_state_dict(state_dict)
    model.eval()
    return model


# ----------------------------
# RECONSTRUCTION LOSS
# ----------------------------

@torch.no_grad()
def compute_recon_mse(model, val_loader, device, is_3d: bool):
    total, n = 0.0, 0

    for batch in val_loader:
        x = batch[0].to(device)
        B = x.shape[0]

        x_in = x if is_3d else x.reshape(B, -1)

        mu, logvar = model.weight_encoder(x_in)
        z = model.weight_encoder._reparameterize(mu, logvar)
        theta = model.weight_encoder.decode_modulations(z)

        recon = model._inr_decode(theta)

        target = x.reshape(B, -1)
        if not is_3d:
            target = target.clamp(-1, 1)

        if recon.shape != target.shape:
            recon = recon.view_as(target)

        loss = F.mse_loss(recon, target, reduction="sum")
        total += loss.item()
        n += B

    return total / n


# ----------------------------
# FID
# ----------------------------

@torch.no_grad()
def compute_fid(model, data_config, n_samples, batch_size, device):
    is_3d = data_config.get("is_3d", False)
    if is_3d:
        return float("nan")

    channels = data_config["channels"]
    img_size = data_config["img_size"]

    inception = _get_inception(device)
    dataset_name = data_config.get("dataset", "").lower()

    if dataset_name == "mnist":
        classifier = _load_classifier(device)
        _, real_feats, _ = _load_or_compute_real_features(
            classifier, inception, device
        )
    else:
        _, real_feats, _ = _load_or_compute_real_features(
            None, inception, device
        )

    samples = []
    remaining = n_samples

    while remaining > 0:
        n = min(batch_size, remaining)
        x = model.sample(n_samples=n)
        x = x.reshape(n, channels, img_size, img_size)
        x = (x * 0.5 + 0.5).clamp(0, 1)
        samples.append(x.cpu())
        remaining -= n

    samples = torch.cat(samples, dim=0)
    gen_feats = _inception_features(samples, inception, device)

    return float(_fid(real_feats, gen_feats))


# ----------------------------
# MAIN
# ----------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config
    with open(args.wd_config) as f:
        config = json.load(f)

    hparams = config

    # dataset
    dataset, val_dataset, data_config = build_dataset(
        dataset_name=hparams["dataset"],
        data_root="data/",
        subset_frac=1.0,
        single_class=False,
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    is_3d = data_config.get("is_3d", False)

    # model
    model = load_model(args.run_dir, config, device, data_config)

    print("\n--- Evaluation ---")

    # recon
    recon_mse = compute_recon_mse(model, val_loader, device, is_3d)
    print(f"Reconstruction MSE: {recon_mse:.6f}")

    # FID / MMD
    if is_3d:
        from src.utility.voxel_metrics import compute_mmd_cov

        all_gen = []
        remaining = args.n_fid_samples

        while remaining > 0:
            n = min(args.fid_batch_size, remaining)
            x = model.sample(n_samples=n)
            all_gen.append(x.cpu())
            remaining -= n

        gen = torch.cat(all_gen, dim=0)
        ref = torch.cat([b[0] for b in val_loader], dim=0)

        mmd, cov = compute_mmd_cov(gen, ref)
        print(f"MMD: {mmd:.6f} | COV: {cov:.6f}")

    else:
        fid = compute_fid(
            model,
            data_config,
            args.n_fid_samples,
            args.fid_batch_size,
            device,
        )
        print(f"FID: {fid:.3f}")

    metrics = {
        "run_name": args.run_dir,
        "recon_mse": recon_mse,
        "fid": fid if not is_3d else None,
        "mmd": mmd if is_3d else None,
        "cov": cov if is_3d else None,
        "n_fid_samples": args.n_fid_samples,
    }

    out_path = os.path.join(args.run_dir, "eval_metrics.json")

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics → {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()