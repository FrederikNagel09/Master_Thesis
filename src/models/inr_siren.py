import math

import torch
import torch.nn as nn

from src.utils.general_utils import get_inr_param_shapes


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


class INRMLP(nn.Module):
    """
    Three-layer SIREN MLP whose weights are supplied at forward time by the
    hypernetwork.  The module itself has NO trainable parameters.

    The INR architecture is:
        SineLayer(2  -> h1, is_first=True)
        SineLayer(h1 -> h2)
        SineLayer(h2 -> h3)
        Linear(h3 -> 1) + Sigmoid
    """

    def __init__(self, h1: int = 20, h2: int = 20, h3: int = 20, omega_0: float = 20.0):
        super().__init__()
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.omega_0 = omega_0
        self.param_shapes = get_inr_param_shapes(h1, h2, h3)

    def _unpack_weights(self, flat: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Slice the flat weight vector output by the hypernetwork into the
        individual weight matrices and bias vectors for each layer.

        Args:
            flat: (total_params,) tensor from the hypernetwork output

        Returns:
            List of (W, b) tuples, one per layer.
        """
        params = []
        offset = 0
        for w_shape, b_shape in self.param_shapes:
            w_size = math.prod(w_shape)
            b_size = math.prod(b_shape)
            w = flat[offset : offset + w_size].view(w_shape)
            offset += w_size
            b = flat[offset : offset + b_size].view(b_shape)
            offset += b_size
            params.append((w, b))
        return params

    def forward(self, coords: torch.Tensor, flat_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords:       (B, N, 2)  — pixel coordinates
            flat_weights: (B, P)     — weights for each image in the batch
        Returns:
            (B, N, 1)
        """
        b, _, _ = coords.shape
        params = self._unpack_weights_batched(flat_weights)  # list of (B, out, in) and (B, out)
        x = coords  # (B, N, 2)

        # Batched linear: x @ W.T + b
        # x: (B, N, in), W: (B, out, in) -> bmm(x, W.transpose) -> (B, N, out)
        for _, (w, b) in enumerate(params[:-1]):
            x = torch.bmm(x, w.transpose(1, 2)) + b.unsqueeze(1)
            x = torch.sin(self.omega_0 * x)

        w, b = params[-1]
        x = torch.bmm(x, w.transpose(1, 2)) + b.unsqueeze(1)
        return x.clamp(0, 1)

    def _unpack_weights_batched(self, flat: torch.Tensor) -> list:
        batch_size = flat.shape[0]
        params = []
        offset = 0
        for w_shape, b_shape in self.param_shapes:
            w_size = math.prod(w_shape)
            b_size = math.prod(b_shape)
            w = flat[:, offset : offset + w_size].view(batch_size, *w_shape)
            offset += w_size
            bias = flat[:, offset : offset + b_size].view(batch_size, *b_shape)
            offset += b_size
            params.append((w, bias))
        return params
