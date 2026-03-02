import math

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """Single SIREN layer: linear + sine activation."""

    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features)

    def _init_weights(self, in_features: int):
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/n, 1/n]
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                # Hidden layers: uniform in [-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0]
                bound = math.sqrt(6.0 / in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenINR(nn.Module):
    """
    Three-layer SIREN MLP for implicit neural representation (INR).
    """

    def __init__(
        self,
        h1: int = 128,
        h2: int = 256,
        h3: int = 128,
        omega_0: float = 30.0,  # 30 is chosen based on paper recommendations.
    ):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(2, h1, omega_0=omega_0, is_first=True),
            SineLayer(h1, h2, omega_0=omega_0, is_first=False),
            SineLayer(h2, h3, omega_0=omega_0, is_first=False),
            nn.Linear(h3, 1),  # Final layer: no sine, just linear
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
