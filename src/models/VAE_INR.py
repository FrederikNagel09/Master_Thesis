import torch
import torch.distributions as td
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812


class VAEINR(nn.Module):
    """
    VAE whose decoder produces INR weights instead of pixel values directly.

    Encoder: image (784,) → q(z|x)
    Decoder: z → flat INR weights
    INR:     coords + weights → pixel predictions
    """

    def __init__(self, prior, encoder, decoder_net, inr, beta=1.0, prior_type="gaussian"):
        """
        prior:        prior distribution p(z)
        encoder:      maps image → q(z|x)
        decoder_net:  nn.Module, maps z → flat weight vector (dim = inr.num_weights)
        inr:          INR instance (stateless forward pass)
        beta:         weight on KL term
        """
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder_net = decoder_net
        self.inr = inr
        self.beta = beta
        self.prior_type = prior_type

    def decode_to_weights(self, z):
        """z: (batch, latent_dim) → flat_weights: (batch, num_weights)"""
        return self.decoder_net(z)

    def elbo(self, image_flat, coords, pixels):
        """
        image_flat: (batch, 784)      — input to encoder
        coords:     (batch, 784, 2)   — pixel coordinates
        pixels:     (batch, 784, 1)   — binary target values
        """
        # Encode image  ( posterior q(z|x) )
        q = self.encoder(image_flat)
        z = q.rsample()  # (batch, latent_dim)

        # Decode sampled z into  INR weights
        flat_weights = self.decode_to_weights(z)  # (batch, num_weights)

        # Run INR: coords + weights = pixel predictions
        pixel_preds = self.inr(coords, flat_weights)  # (batch, 784, 1)

        # Use MSE for reconstruction loss
        recon_loss = F.binary_cross_entropy(pixel_preds, pixels, reduction="sum") / image_flat.size(0)

        # Compute KL divergence between q(z|x) and p(z)
        kl = td.kl_divergence(q, self.prior()).mean()

        # print(f"  [DEBUG] recon_loss: {recon_loss.item():.4f}  kl: {kl.item():.4f}")

        elbo = -(recon_loss + self.beta * kl)
        return elbo, recon_loss, kl

    def sample(self, coords, n_samples=1):
        """
        Sample INR weights from the prior, then render at given coords.
        coords: (784, 2) or (n_samples, 784, 2)
        """
        # Prior sampling:
        z = self.prior().sample(torch.Size([n_samples]))  # (n_samples, latent_dim)

        # Decode z into INR weights
        flat_weights = self.decode_to_weights(z)

        # Throw weights and given pixel grid (coords) into the INR to get pixel predictions
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(n_samples, -1, -1)
        return self.inr(coords, flat_weights)  # (n_samples, 784, 1)

    def elbo_mog(self, image_flat, coords, pixels, n_samples=10):
        """
        Compute the ELBO for the given batch of data with a MoG prior.

        image_flat: (batch, 784)      — input to encoder
        coords:     (batch, 784, 2)   — pixel coordinates
        pixels:     (batch, 784, 1)   — binary target values
        n_samples:  [int]             — MC samples for KL estimation
        """
        # 1. Encode image → posterior q(z|x)
        q = self.encoder(image_flat)
        z = q.rsample()  # (batch, latent_dim)

        # 2. Decode z → INR weights → pixel predictions
        flat_weights = self.decode_to_weights(z)
        pixel_preds = self.inr(coords, flat_weights)  # (batch, 784, 1)

        # 3. Reconstruction loss
        recon_loss = F.binary_cross_entropy(pixel_preds, pixels, reduction="sum") / image_flat.size(0)

        # 4. MC estimation of KL divergence
        kl_divergence = 0
        for _ in range(n_samples):
            z_sample = q.rsample()
            log_p_z = self.prior().log_prob(z_sample)  # MoG log prob
            log_q_z_x = q.log_prob(z_sample)  # encoder log prob
            kl_divergence += log_q_z_x - log_p_z
        kl_divergence = kl_divergence / n_samples  # (batch,)

        # 5. ELBO
        kl = kl_divergence.mean()
        elbo = -(recon_loss + self.beta * kl)
        return elbo, recon_loss, kl

    def forward(self, image_flat, coords, pixels):
        if self.prior_type == "mog":
            loss, recon, kl = self.elbo_mog(image_flat, coords, pixels)
        else:
            loss, recon, kl = self.elbo(image_flat, coords, pixels)

        # torch.tensor(0.0) - because theres no l_diff term in this model, but we want to return 4 values
        # for consistency with NDM and NDM-INR
        return -loss, torch.tensor(0.0), kl, recon
