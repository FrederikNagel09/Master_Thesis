import torch
import torch.distributions as td
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812


class INR(nn.Module):
    """
    A small MLP that predicts pixel values from (x,y) coordinates.
    Weights are supplied externally by the hypernetwork (VAE decoder).
    """

    def __init__(self, coord_dim=2, hidden_dim=20, n_hidden=2, out_dim=1):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.out_dim = out_dim

        # Compute the total number of weights this INR needs
        dims = [coord_dim] + [hidden_dim] * n_hidden + [out_dim]
        self.weight_shapes = []
        self.bias_shapes = []
        total = 0
        for i in range(len(dims) - 1):
            self.weight_shapes.append((dims[i + 1], dims[i]))
            self.bias_shapes.append((dims[i + 1],))
            total += dims[i + 1] * dims[i] + dims[i + 1]
        self.num_weights = total

    def forward(self, coords, flat_weights):
        """
        coords:       (batch, n_pixels, 2)
        flat_weights: (batch, num_weights)
        returns:      (batch, n_pixels, 1)  — logits
        """
        batch = coords.shape[0]
        idx = 0
        x = coords  # (batch, n_pixels, coord_dim)

        for i, (ws, _bs) in enumerate(zip(self.weight_shapes, self.bias_shapes)):  # noqa: B905
            out_f, in_f = ws
            W = flat_weights[:, idx : idx + out_f * in_f].view(batch, out_f, in_f)  # noqa: N806
            idx += out_f * in_f
            b = flat_weights[:, idx : idx + out_f].view(batch, 1, out_f)
            idx += out_f

            # (batch, n_pixels, out_f)
            x = torch.bmm(x, W.transpose(1, 2)) + b

            # ReLU on hidden layers, sigmoid on last
            x = torch.relu(x) if i < len(self.weight_shapes) - 1 else torch.sigmoid(x)

        return x  # (batch, n_pixels, 1)


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
        encoder:      maps image → q(z|x)  (your existing encoder module)
        decoder_net:  nn.Module, maps z → flat weight vector (dim = inr.num_weights)
        inr:          INR instance (stateless forward pass)
        beta:         weight on KL term (beta-VAE style, use 1.0 for standard VAE)
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
        # 1. Encode image → posterior q(z|x)
        q = self.encoder(image_flat)
        z = q.rsample()  # (batch, latent_dim)

        # 2. Decode z → INR weights
        flat_weights = self.decode_to_weights(z)  # (batch, num_weights)

        # 3. Run INR: coords + weights → pixel predictions
        pixel_preds = self.inr(coords, flat_weights)  # (batch, 784, 1)

        # 4. Reconstruction loss: BCE since pixels are binary
        recon_loss = F.binary_cross_entropy(pixel_preds, pixels, reduction="mean")

        # 5. KL divergence
        kl = td.kl_divergence(q, self.prior()).mean()

        elbo = -(recon_loss + self.beta * kl)
        return elbo, recon_loss, kl

    def sample(self, coords, n_samples=1):
        """
        Sample INR weights from the prior, then render at given coords.
        coords: (784, 2) or (n_samples, 784, 2)
        """
        z = self.prior().sample(torch.Size([n_samples]))  # (n_samples, latent_dim)
        flat_weights = self.decode_to_weights(z)
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
        recon_loss = F.binary_cross_entropy(pixel_preds, pixels, reduction="mean")

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
        return -loss, recon, kl
