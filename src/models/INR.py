import math

import torch
import torch.nn as nn


class INR(nn.Module):
    """
    A small MLP that predicts pixel values from (x,y) coordinates.
    Weights are supplied externally by the hypernetwork (VAE decoder).
    """

    def __init__(self, coord_dim=2, hidden_dim=20, n_hidden=2, out_dim=1, output_activation="sigmoid"):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.out_dim = out_dim
        self.output_activation = output_activation

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
            if i < len(self.weight_shapes) - 1:
                x = torch.relu(x)
            else:
                x = torch.sigmoid(x) if self.output_activation == "sigmoid" else torch.tanh(x)

        return x  # (batch, n_pixels, 1)


class SirenINR(nn.Module):
    """
    A SIREN (Sinusoidal Representation Network) INR that predicts pixel
    values from (x,y) coordinates. Weights are supplied externally by the
    hypernetwork (VAE decoder / NDM encoder), identical interface to INR.

    Each hidden layer applies: sin(omega_0 * (W x + b))
    The first layer uses a wider initialisation range per the SIREN paper.

    Reference: Sitzmann et al., "Implicit Neural Representations with
    Periodic Activation Functions", NeurIPS 2020.
    """

    def __init__(
        self,
        coord_dim: int = 2,
        hidden_dim: int = 20,
        n_hidden: int = 2,
        out_dim: int = 1,
        output_activation: str = "sigmoid",
        omega_0: float = 30.0,  # frequency for hidden layers
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.out_dim = out_dim
        self.output_activation = output_activation
        self.omega_0 = omega_0
        print(f"  Initialising SirenINR with omega_0 = {omega_0}")

        # ── Layer dimensions ──────────────────────────────────────────────────
        dims = [coord_dim] + [hidden_dim] * n_hidden + [out_dim]

        self.weight_shapes = []
        self.bias_shapes = []
        total = 0
        for i in range(len(dims) - 1):
            self.weight_shapes.append((dims[i + 1], dims[i]))
            self.bias_shapes.append((dims[i + 1],))
            total += dims[i + 1] * dims[i] + dims[i + 1]
        self.num_weights = total

        # ── SIREN initialisation bounds (stored, not used as parameters) ──────
        # Used by the hypernetwork to initialise its output layer correctly.
        # First layer: uniform(-1/in_f, 1/in_f)
        # Hidden layers: uniform(-sqrt(6/in_f)/omega_0, sqrt(6/in_f)/omega_0)
        self.init_schemes = []
        for i, (_out_f, in_f) in enumerate(ws for ws in self.weight_shapes):
            bound = 1.0 / in_f if i == 0 else math.sqrt(6.0 / in_f) / omega_0
            self.init_schemes.append(bound)

    def forward(self, coords: torch.Tensor, flat_weights: torch.Tensor) -> torch.Tensor:
        """
        coords:       (batch, n_pixels, coord_dim)
        flat_weights: (batch, num_weights)
        returns:      (batch, n_pixels, out_dim)
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

            if i < len(self.weight_shapes) - 1:
                # Hidden layer: sinusoidal activation scaled by omega_0
                x = torch.sin(self.omega_0 * x)
            else:
                # Output layer: standard activation (no sin)
                if self.output_activation == "sigmoid":
                    x = torch.sigmoid(x)
                elif self.output_activation == "tanh":
                    x = torch.tanh(x)
                # "none" / anything else → linear output

        return x  # (batch, n_pixels, out_dim)
