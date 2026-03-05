import torch
import torch.distributions as td
import torch.nn as nn
import torch.utils.data


class GaussianPrior(nn.Module):
    def __init__(self, latent_dim):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        latent_dim: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()  # noqa: UP008
        self.M = latent_dim
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MoGPrior(nn.Module):
    def __init__(self, latent_dim, gauss_components=10):
        """
        Define a Mixture of Gaussian (MoG) prior with K components.

        Parameters:
        latent_dim: [int]  -> Dimension of the latent space.
        gauss_components: [int]  -> Number of Gaussian components in the mixture.
        """
        super(MoGPrior, self).__init__()  # noqa: UP008
        self.latent_dim = latent_dim
        self.K = gauss_components

        # Learnable parameters: mixture weights, means, and standard deviations
        self.mixture_weights = nn.Parameter(torch.ones(gauss_components) / gauss_components, requires_grad=True)
        self.means = nn.Parameter(torch.randn(gauss_components, latent_dim), requires_grad=True)
        self.stds = nn.Parameter(torch.ones(gauss_components, latent_dim), requires_grad=True)

    def forward(self):
        """
        Return the Mixture of Gaussians prior distribution.

        Returns:
        prior: [torch.distributions.Distribution] -> MoG prior
        """
        # Create K Gaussian distributions
        components = td.Independent(td.Normal(loc=self.means, scale=self.stds.exp()), 1)

        # Convert logits to probabilities
        mixture_dist = td.Categorical(logits=self.mixture_weights)

        # Define the Mixture of Gaussians
        mog_prior = td.MixtureSameFamily(mixture_dist, components)

        return mog_prior


class VAMPPrior(nn.Module):
    def __init__(self, encoders, input_shape, num_components):
        """
        Variational Mixture of Posterior Prior (VAMP Prior)

        Parameters:
        encoders: list of GaussianEncoder instances
            List of approximate posterior networks.
        input_shape: tuple
            Shape of the pseudo-inputs (e.g., (1, 28, 28) for MNIST).
        num_components: int
            Number of pseudo-inputs (mixture components).
        """
        super(VAMPPrior, self).__init__()  # noqa: UP008
        self.encoders = encoders
        self.num_components = num_components

        # Learnable pseudo-inputs in the data space
        self.pseudo_inputs = nn.Parameter(torch.randn(num_components, *input_shape) * 0.1)

    def forward(self):
        """
        Compute the VAMP prior using the approximate posterior networks.
        """
        means_list, stds_list = [], []

        # Pass pseudo-inputs through each encoder
        for encoder in self.encoders:
            posterior = encoder(self.pseudo_inputs)  # Returns Independent(Normal)
            means_list.append(posterior.base_dist.loc)  # Shape: [num_components, latent_dim]
            stds_list.append(posterior.base_dist.scale)  # Shape: [num_components, latent_dim]

        # Stack results across encoders (concatenating over mixture components)
        means = torch.cat(means_list, dim=0)  # Shape: [num_components * num_encoders, latent_dim]
        stds = torch.cat(stds_list, dim=0)  # Shape: [num_components * num_encoders, latent_dim]

        # Define mixture components
        components = td.Independent(td.Normal(loc=means, scale=stds), 1)

        # Define uniform mixture distribution
        mixture_dist = td.Categorical(probs=torch.ones(means.shape[0]) / means.shape[0])

        # Create VAMP prior
        return td.MixtureSameFamily(mixture_dist, components)
