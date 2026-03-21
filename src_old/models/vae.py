import torch
import torch.distributions as td
import torch.nn as nn
from torch.nn import functional as F  # noqa: F401, N812


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder, type="normal"):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()  # noqa: UP008
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.MoG = type

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        # Encoder computes the approximate posterior q(z|x)
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()))
        return elbo

    def elbo_mog(self, x, n_samples=10):
        """
        Compute the ELBO for the given batch of data with a MoG prior.

        Parameters:
        x: [torch.Tensor]
        A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
        n_samples: [int]
        Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        # Sample from q(z|x)
        q = self.encoder(x)
        z = q.rsample()

        # Compute the reconstruction loss
        recon_log_prob = self.decoder(z).log_prob(x)

        # Monte Carlo estimation of KL div
        kl_divergence = 0
        for _ in range(n_samples):
            # Sample from q(z|x)
            z_sample = q.rsample()

            # prior
            mog_prior = self.prior()
            log_p_z = mog_prior.log_prob(z_sample)
            # aggregated posterior
            log_q_z_x = q.log_prob(z_sample)

            # Compute KL
            kl_divergence += log_q_z_x - log_p_z

        # Average of KL div Approximation
        kl_divergence = kl_divergence / n_samples

        # Compute the ELBO:
        elbo = recon_log_prob.mean() - kl_divergence.mean()

        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        if self.MoG == "mog":
            return -self.elbo_mog(x)
        else:
            return -self.elbo(x)
