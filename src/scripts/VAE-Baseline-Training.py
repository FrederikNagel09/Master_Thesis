import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")

from src.models.trans_inr import TransInr, make_coord_grid

# Assuming these relative imports align with your repository structure
from src.utility.dataset_builders import build_dataset

# ──────────────────────────────────────────────────────────────────────────────
# HARDCODED HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
RUN_NAME = "vae-test"
DATASET_NAME = "mnist"
EPOCHS = 1
BATCH_SIZE = 128
LR = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SUBSET_FRAC = 0.2
NORMALIZE = True

# VAE Latent Settings
LATENT_DIM = 32
LATENT_SIZE = 4  # Results in (4, 4) spatial latent representation
LATENT_PATCH_SIZE = 2
LATENT_ENC_HIDDEN_DIM = 12  # Maps to hidden_dim in the encoder

# Decoder (TransInr) Settings
DEC_TRANS_DIM = 128
DEC_TRANS_N_HEAD = 8
DEC_TRANS_HEAD_DIM = 32
DEC_TRANS_FF_DIM = 1024
DEC_TRANS_ENC_DEPTH = 4
DEC_TRANS_DEC_DEPTH = 4
DEC_TRANS_N_GROUPS = 32
DEC_TRANS_UPDATE_STRATEGY = "scale"
INR_HIDDEN_DIM = 128
INR_LAYERS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")


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
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels)
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

        ch1 = hidden_dim
        ch2 = hidden_dim * 2
        ch3 = hidden_dim * 4
        ch4 = hidden_dim * 8

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch1, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(ch1), nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_stage(ch1, ch1, num_blocks=2, stride=1)
        self.layer2 = self._make_stage(ch1, ch2, num_blocks=2, stride=2)
        self.layer3 = self._make_stage(ch2, ch3, num_blocks=2, stride=2)
        self.layer4 = self._make_stage(ch3, ch4, num_blocks=2, stride=2)

        # Separate learnable heads for the distribution parameters
        self.upsample_mu = nn.ConvTranspose2d(in_channels=ch4, out_channels=latent_dim, kernel_size=4, stride=2, padding=1)
        self.upsample_logvar = nn.ConvTranspose2d(in_channels=ch4, out_channels=latent_dim, kernel_size=4, stride=2, padding=1)

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

        # Emulating your system's coordinate grid processing layout
        coord_grid = make_coord_grid((img_size, img_size), (-1, 1))  # Shape: (H, W, 2)
        self.register_buffer("coord_grid", coord_grid)
        self.device = device

    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Emulates the signature contract used by your plotting routines
        """
        # Mirroring how TransInr parses latents vs spatial positions
        # Standard layout passes batch elements or updates dynamically via coordinate context
        batch_size = z.shape[0]
        coords = self.coord_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.device)

        # Calls the TransInr functional forward path
        return self.decoder(z, coords)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.latent_encoder(x)
        z = self.latent_encoder.reparameterize(mu, logvar)
        x_recon = self._decode_latent(z)
        return x_recon, mu, logvar


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING WORKFLOW
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"--- Initialization Process Started: {RUN_NAME} ---")

    # 1. Build Dataset
    class ArgsMock:
        dataset = DATASET_NAME
        subset_frac = SUBSET_FRAC

    dataset, data_config = build_dataset(
        dataset_name=ArgsMock.dataset,
        data_root="data/",
        subset_frac=ArgsMock.subset_frac,
        single_class=False,
        single_class_label=1,
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    channels = data_config["channels"]
    img_size = data_config["img_size"]
    latent_size_tuple = (LATENT_SIZE, LATENT_SIZE)

    # 2. Build Structural Subnetworks
    encoder = ProbabilisticResNetLatentEncoder(
        in_channels=channels,
        latent_dim=LATENT_DIM,
        latent_size=latent_size_tuple,
        hidden_dim=LATENT_ENC_HIDDEN_DIM,
    )

    tokenizer_cfg = {
        "target": "src.models.trans_inr_helpers.LatentTokenizer",
        "params": {
            "latent_dim": LATENT_DIM,
            "latent_size": LATENT_SIZE,
            "patch_size": LATENT_PATCH_SIZE,
            "dim": DEC_TRANS_DIM,
            "n_head": DEC_TRANS_N_HEAD,
            "head_dim": DEC_TRANS_HEAD_DIM,
        },
    }
    inr_cfg = {
        "target": "src.models.trans_inr_helpers.SIREN",
        "params": {
            "depth": INR_LAYERS,
            "in_dim": 2,
            "out_dim": channels,
            "hidden_dim": INR_HIDDEN_DIM,
            "out_bias": 0.5,
        },
    }
    transformer_cfg = {
        "target": "src.models.trans_inr_helpers.Transformer",
        "params": {
            "dim": DEC_TRANS_DIM,
            "encoder_depth": DEC_TRANS_ENC_DEPTH,
            "decoder_depth": DEC_TRANS_DEC_DEPTH,
            "n_head": DEC_TRANS_N_HEAD,
            "head_dim": DEC_TRANS_HEAD_DIM,
            "ff_dim": DEC_TRANS_FF_DIM,
        },
    }

    decoder = TransInr(
        tokenizer=tokenizer_cfg,
        inr=inr_cfg,
        data_shape=(img_size, img_size),
        n_groups=DEC_TRANS_N_GROUPS,
        transformer=transformer_cfg,
        update_strategy=DEC_TRANS_UPDATE_STRATEGY,
    )

    # Wrap up into VAE
    model = VAEWrapper(encoder, decoder, img_size, DEVICE).to(DEVICE)

    # 3. Setup Optimization Components
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"Training initialized on {DEVICE} for {EPOCHS} epochs.")

    # 4. Core Optimization Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_mse = 0.0
        running_kl = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")

        for batch in progress_bar:
            # Handle standard dataset returns where index 0 holds the image tensors
            x = batch[0].to(DEVICE)

            # Match reconstruction flattening rules if dataset returns flattened vectors
            if x.dim() == 2:
                x = x.view(x.shape[0], channels, img_size, img_size)

            optimizer.zero_grad()

            # Forward Pass
            x_recon, mu, logvar = model(x)

            # Flatten targets and predictions to compute sum of squared errors per sample
            x_hat_flat = x_recon.reshape(x_recon.shape[0], -1)
            x_flat = x.reshape(x.shape[0], -1).clamp(-1, 1)

            # Scaled sum-squared error matching your exact formulation
            loss_mse = 0.5 * ((x_flat - x_hat_flat) ** 2).sum(dim=-1).mean()

            # Analytical KL Divergence adjusted for the sum scale
            loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]))

            # Total Loss
            kl_weight = 0.1
            total_loss = loss_mse + (kl_weight * loss_kl)

            # Optimization step
            total_loss.backward()
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            # Logging tracking metrics
            running_mse += loss_mse.item()
            running_kl += loss_kl.item()

            progress_bar.set_postfix({"MSE": f"{loss_mse.item():.4f}", "KL": f"{loss_kl.item():.2f}"})

        epoch_mse = running_mse / len(dataloader)
        epoch_kl = running_kl / len(dataloader)
        print(f"      ↳ [Summary] Avg MSE: {epoch_mse:.5f} | Avg KL: {epoch_kl:.3f}")

    print("--- Training Execution Finished Successful ---")

    print("--- Generating 5x5 Grid of Random Samples ---")
    import torchvision.utils as vutils

    model.eval()
    os.makedirs("src/results", exist_ok=True)

    with torch.no_grad():
        # Standard Gaussian prior samples matching spatial latent shape (5x5 grid = 25 images)
        z_random = torch.randn(25, LATENT_DIM, LATENT_SIZE, LATENT_SIZE).to(DEVICE)

        # Decode samples using your wrapper's contract path
        samples = model._decode_latent(z_random)

        # Denormalize to [0, 1] range if your data operates on standard [-1, 1] scales
        samples = (samples * 0.5 + 0.5).clamp(0, 1)

        # Create grid and save to disk
        vutils.save_image(samples, "src/results/vae_samples.png", nrow=5, padding=2)
        print("Sample grid successfully saved to src/results/vae_samples.png")


if __name__ == "__main__":
    main()
