import torch
import torch.distributions as td
import torch.nn as nn


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
