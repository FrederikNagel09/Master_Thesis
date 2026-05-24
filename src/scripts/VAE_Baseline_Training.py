import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")
from src.models.trans_inr import TransInr, make_coord_grid
from src.utility.dataset_builders import build_dataset

"""
python src/scripts/VAE_Baseline_Training.py \
    --run_name vae-test \
    --dataset mnist \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --subset_frac 0.1 \
    --lambda_kl_max 0.01 \
    --kl_warmup_frac 0.4 \
    --latent_dim 32 \
    --latent_size 4 \
    --latent_patch_size 2 \
    --latent_enc_hidden_dim 12 \
    --dec_trans_dim 128 \
    --dec_trans_n_head 8 \
    --dec_trans_head_dim 32 \
    --dec_trans_ff_dim 1024 \
    --dec_trans_enc_depth 4 \
    --dec_trans_dec_depth 4 \
    --dec_trans_n_groups 32 \
    --dec_trans_update_strategy scale \
    --inr_hidden_dim 128 \
    --inr_layers 3
"""

# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments, all defaulting to the original hardcoded hyperparameters.

    Returns:
        argparse.Namespace: parsed arguments
    """
    p = argparse.ArgumentParser(description="Train a TransINR-VAE model")

    # Run
    p.add_argument("--run_name", type=str, default="vae-test")
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--results_dir", type=str, default="src/results")

    # Training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--subset_frac", type=float, default=0.2)

    # KL
    p.add_argument("--lambda_kl_max", type=float, default=0.1, help="Maximum KL weight after warm-up")
    p.add_argument("--kl_warmup_frac", type=float, default=0.4, help="Fraction of total epochs over which KL ramps from 0 to lambda_kl_max")

    # Encoder
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--latent_size", type=int, default=4)
    p.add_argument("--latent_patch_size", type=int, default=2)
    p.add_argument("--latent_enc_hidden_dim", type=int, default=12)

    # Decoder (TransInr)
    p.add_argument("--dec_trans_dim", type=int, default=128)
    p.add_argument("--dec_trans_n_head", type=int, default=8)
    p.add_argument("--dec_trans_head_dim", type=int, default=32)
    p.add_argument("--dec_trans_ff_dim", type=int, default=1024)
    p.add_argument("--dec_trans_enc_depth", type=int, default=4)
    p.add_argument("--dec_trans_dec_depth", type=int, default=4)
    p.add_argument("--dec_trans_n_groups", type=int, default=32)
    p.add_argument("--dec_trans_update_strategy", type=str, default="scale")

    # INR
    p.add_argument("--inr_hidden_dim", type=int, default=128)
    p.add_argument("--inr_layers", type=int, default=3)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# PROBABILISTIC RESNET LATENT ENCODER
# ──────────────────────────────────────────────────────────────────────────────


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ProbabilisticResNetLatentEncoder(nn.Module):
    """
    Encodes an image to a probabilistic latent feature space (mu, logvar)
    using a scalable ResNet backbone.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        latent_size: tuple[int, int],
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.latent_size = latent_size if isinstance(latent_size, tuple) else (latent_size, latent_size)

        ch1, ch2, ch3, ch4 = hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_stage(ch1, ch1, num_blocks=2, stride=1)
        self.layer2 = self._make_stage(ch1, ch2, num_blocks=2, stride=2)
        self.layer3 = self._make_stage(ch2, ch3, num_blocks=2, stride=2)
        self.layer4 = self._make_stage(ch3, ch4, num_blocks=2, stride=2)

        self.upsample_mu = nn.ConvTranspose2d(ch4, latent_dim, kernel_size=4, stride=2, padding=1)
        self.upsample_logvar = nn.ConvTranspose2d(ch4, latent_dim, kernel_size=4, stride=2, padding=1)

    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNetBasicBlock(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        mu = self.upsample_mu(out)
        logvar = self.upsample_logvar(out)
        if mu.shape[-2:] != self.latent_size:
            mu = nn.functional.interpolate(mu, size=self.latent_size, mode="bilinear", align_corners=False)
            logvar = nn.functional.interpolate(logvar, size=self.latent_size, mode="bilinear", align_corners=False)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu


# ──────────────────────────────────────────────────────────────────────────────
# VAE SYSTEM WRAPPER
# ──────────────────────────────────────────────────────────────────────────────


class VAEWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, img_size: int, device: torch.device):
        super().__init__()
        self.latent_encoder = encoder
        self.decoder = decoder
        self.img_size = img_size
        self.device = device

        coord_grid = make_coord_grid((img_size, img_size), (-1, 1))  # (H, W, 2)
        self.register_buffer("coord_grid", coord_grid)

    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes a latent tensor through the TransInr decoder."""
        batch_size = z.shape[0]
        coords = self.coord_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.device)
        return self.decoder(z, coords)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.latent_encoder(x)
        z = self.latent_encoder.reparameterize(mu, logvar)
        x_recon = self._decode_latent(z)
        return x_recon, mu, logvar


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────


def save_training_graph(
    history: dict[str, list[float]],
    steps_per_epoch: int,
    epochs: int,
    save_path: str,
) -> None:
    """
    Saves a 3-panel training graph (total ELBO, recon loss, KL loss) with
    per-step lines and epoch-level x-axis ticks.

    Args:
        history:          dict with keys "elbo", "recon", "kl", each a list of per-step values
        steps_per_epoch:  number of optimizer steps per epoch
        epochs:           total number of epochs trained
        save_path:        full file path to save the .png
    """
    total_steps = len(history["elbo"])

    # Sample ~10 evenly spaced ticks regardless of epoch count
    max_ticks = 10
    step = max(1, epochs // max_ticks)
    tick_positions = [i * steps_per_epoch for i in range(0, epochs + 1, step)]
    tick_labels = [str(i) for i in range(0, epochs + 1, step)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("elbo", "Total ELBO", "tab:blue"),
        ("recon", "Reconstruction Loss", "tab:orange"),
        ("kl", "KL Loss", "tab:green"),
    ]

    for ax, (key, title, color) in zip(axes, panels):  # noqa: B905
        ax.plot(range(total_steps), history[key], color=color, linewidth=0.8, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training graph saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MODEL SAVING
# ──────────────────────────────────────────────────────────────────────────────


def save_model(model: nn.Module, args: argparse.Namespace, results_dir: str) -> None:
    """
    Saves model weights (state_dict) and config (JSON) to results_dir.

    Args:
        model:       trained VAEWrapper
        args:        parsed CLI args used to build the model
        results_dir: directory to save into
    """
    weights_path = os.path.join(results_dir, f"{args.run_name}_weights.pt")
    config_path = os.path.join(results_dir, f"{args.run_name}_config.json")

    # Weights
    torch.save(model.state_dict(), weights_path)

    # Config — everything needed to reconstruct the model architecture
    config = {
        "run_name": args.run_name,
        "dataset": args.dataset,
        "latent_dim": args.latent_dim,
        "latent_size": args.latent_size,
        "latent_patch_size": args.latent_patch_size,
        "latent_enc_hidden_dim": args.latent_enc_hidden_dim,
        "dec_trans_dim": args.dec_trans_dim,
        "dec_trans_n_head": args.dec_trans_n_head,
        "dec_trans_head_dim": args.dec_trans_head_dim,
        "dec_trans_ff_dim": args.dec_trans_ff_dim,
        "dec_trans_enc_depth": args.dec_trans_enc_depth,
        "dec_trans_dec_depth": args.dec_trans_dec_depth,
        "dec_trans_n_groups": args.dec_trans_n_groups,
        "dec_trans_update_strategy": args.dec_trans_update_strategy,
        "inr_hidden_dim": args.inr_hidden_dim,
        "inr_layers": args.inr_layers,
        "lambda_kl_max": args.lambda_kl_max,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model weights saved to {weights_path}")
    print(f"Model config  saved to {config_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING WORKFLOW
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    print(f"--- Initialization Process Started: {args.run_name} ---")

    # 1. Dataset
    dataset, data_config = build_dataset(
        dataset_name=args.dataset,
        data_root="data/",
        subset_frac=args.subset_frac,
        single_class=False,
        single_class_label=1,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    channels = data_config["channels"]
    img_size = data_config["img_size"]

    # 2. Build Model
    encoder = ProbabilisticResNetLatentEncoder(
        in_channels=channels,
        latent_dim=args.latent_dim,
        latent_size=(args.latent_size, args.latent_size),
        hidden_dim=args.latent_enc_hidden_dim,
    )

    decoder = TransInr(
        tokenizer={
            "target": "src.models.trans_inr_helpers.LatentTokenizer",
            "params": {
                "latent_dim": args.latent_dim,
                "latent_size": args.latent_size,
                "patch_size": args.latent_patch_size,
                "dim": args.dec_trans_dim,
                "n_head": args.dec_trans_n_head,
                "head_dim": args.dec_trans_head_dim,
            },
        },
        inr={
            "target": "src.models.trans_inr_helpers.SIREN",
            "params": {
                "depth": args.inr_layers,
                "in_dim": 2,
                "out_dim": channels,
                "hidden_dim": args.inr_hidden_dim,
                "out_bias": 0.5,
            },
        },
        data_shape=(img_size, img_size),
        n_groups=args.dec_trans_n_groups,
        transformer={
            "target": "src.models.trans_inr_helpers.Transformer",
            "params": {
                "dim": args.dec_trans_dim,
                "encoder_depth": args.dec_trans_enc_depth,
                "decoder_depth": args.dec_trans_dec_depth,
                "n_head": args.dec_trans_n_head,
                "head_dim": args.dec_trans_head_dim,
                "ff_dim": args.dec_trans_ff_dim,
            },
        },
        update_strategy=args.dec_trans_update_strategy,
    )

    model = VAEWrapper(encoder, decoder, img_size, device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # KL warm-up: ramp over first kl_warmup_frac of total epochs
    kl_warmup_epochs = max(1, int(args.kl_warmup_frac * args.epochs))

    print(f"Training on {device} | {args.epochs} epochs | KL warm-up over {kl_warmup_epochs} epochs")

    # Per-step loss history for the training graph
    history = {"elbo": [], "recon": [], "kl": []}

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()

        # KL annealing weight — linearly ramps from 0 → lambda_kl_max
        lambda_kl = args.lambda_kl_max * min(1.0, epoch / kl_warmup_epochs)

        running_mse = 0.0
        running_kl = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

        for batch in progress_bar:
            x = batch[0].to(device)
            if x.dim() == 2:
                x = x.view(x.shape[0], channels, img_size, img_size)

            optimizer.zero_grad()

            x_recon, mu, logvar = model(x)

            x_hat_flat = x_recon.reshape(x_recon.shape[0], -1)
            x_flat = x.reshape(x.shape[0], -1).clamp(-1, 1)

            loss_recon = 0.5 * ((x_flat - x_hat_flat) ** 2).sum(dim=-1).mean()
            loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]))
            total_loss = loss_recon + lambda_kl * loss_kl

            total_loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # Record per-step values
            history["elbo"].append(total_loss.item())
            history["recon"].append(loss_recon.item())
            history["kl"].append(loss_kl.item())

            running_mse += loss_recon.item()
            running_kl += loss_kl.item()
            progress_bar.set_postfix(
                {
                    "MSE": f"{loss_recon.item():.4f}",
                    "KL": f"{loss_kl.item():.2f}",
                    "λ_kl": f"{lambda_kl:.3f}",
                }
            )

        epoch_mse = running_mse / len(dataloader)
        epoch_kl = running_kl / len(dataloader)
        print(f"      ↳ [Summary] Avg MSE: {epoch_mse:.5f} | Avg KL: {epoch_kl:.3f} | λ_kl: {lambda_kl:.4f}")

    # 4. Save artefacts
    os.makedirs(args.results_dir, exist_ok=True)

    # Training graph
    save_training_graph(
        history=history,
        steps_per_epoch=len(dataloader),
        epochs=args.epochs,
        save_path=os.path.join(args.results_dir, f"{args.run_name}_training_curves.png"),
    )

    # Model weights + config
    save_model(model, args, args.results_dir)

    # Sample grid
    import torchvision.utils as vutils

    model.eval()
    with torch.no_grad():
        z_random = torch.randn(25, args.latent_dim, args.latent_size, args.latent_size).to(device)
        samples = model._decode_latent(z_random)
        samples = (samples * 0.5 + 0.5).clamp(0, 1)
        sample_path = os.path.join(args.results_dir, f"{args.run_name}_samples.png")
        vutils.save_image(samples, sample_path, nrow=5, padding=2)
        print(f"Sample grid saved to {sample_path}")

    print("--- Training Execution Finished Successfully ---")


if __name__ == "__main__":
    main()
