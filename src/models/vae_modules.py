import torch
import torch.distributions as td
import torch.nn as nn


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


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super().__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class GaussianFullEncoder(nn.Module):
    """
    Network and forward method for a Gaussian encoder distribution.
    The network is a simple MLP that takes as input a tensor of dimension `(batch_size, feature_dim1, feature_dim2)`,
    and outputs a tensor of dimension `(batch_size, latent_dim * 2)`.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = [nn.Flatten()]
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU()]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, latent_dim * 2))
        self.encoder_net = nn.Sequential(*layers)

    def forward(self, x):
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(log_std)), 1)


class BernoulliFullDecoder(nn.Module):
    """
    Network and forward method for a Bernoulli decoder distribution.
    The network is a simple MLP that takes as input a tensor of dimension `(batch_size, M)`,
    where M is the dimension of the latent space, and outputs a tensor of dimension (batch_size, feature_dim1, feature_dim2).
    The forward method takes as input a tensor of dimension `(batch_size, M)` and returns a Bernoulli distribution over the data space.
    """

    def __init__(self, latent_dim: int, output_shape: tuple[int, int], hidden_dims: list[int]):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU()]
            in_dim = h_dim
        output_dim = output_shape[0] * output_shape[1]
        layers += [nn.Linear(in_dim, output_dim), nn.Unflatten(-1, output_shape)]
        self.decoder_net = nn.Sequential(*layers)

    def forward(self, z):
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)
