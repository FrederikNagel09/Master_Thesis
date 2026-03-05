import torch
import torch.distributions as td
import torch.nn as nn


class GaussianEncoder(nn.Module):
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


class BernoulliDecoder(nn.Module):
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
