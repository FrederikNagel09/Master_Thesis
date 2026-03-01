import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers to compute INR parameter count and shapes
# ---------------------------------------------------------------------------


def get_inr_param_shapes(h1: int, h2: int, h3: int) -> list[tuple[tuple, tuple]]:
    """
    Returns a list of (weight_shape, bias_shape) for each layer of the INR:
        Linear(2  -> h1)
        Linear(h1 -> h2)
        Linear(h2 -> h3)
        Linear(h3 -> 1)
    """
    dims = [2, h1, h2, h3, 1]
    return [((dims[i + 1], dims[i]), (dims[i + 1],)) for i in range(len(dims) - 1)]


def count_inr_params(h1: int, h2: int, h3: int) -> int:
    """Total number of scalar parameters in the INR."""
    total = 0
    for w_shape, b_shape in get_inr_param_shapes(h1, h2, h3):
        total += math.prod(w_shape) + math.prod(b_shape)
    return total


# ---------------------------------------------------------------------------
# SIREN INR — forward-only, weights supplied externally
# ---------------------------------------------------------------------------


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
        """
        flat: (B, P) -> list of (W:(B, out, in), b:(B, out)) per layer
        """
        b = flat.shape[0]
        params = []
        offset = 0
        for w_shape, b_shape in self.param_shapes:
            w_size = math.prod(w_shape)
            b_size = math.prod(b_shape)
            w = flat[:, offset : offset + w_size].view(b, *w_shape)
            offset += w_size
            b = flat[:, offset : offset + b_size].view(b, *b_shape)
            offset += b_size
            params.append((w, b))
        return params


# ---------------------------------------------------------------------------
# HyperNetwork — image → flat INR weights
# ---------------------------------------------------------------------------


class HyperNetwork(nn.Module):
    """
    A simple 3-layer MLP that takes a flattened MNIST image (784 values) and
    outputs a flat vector of weights for the INRMLP.

    Architecture:
        Linear(784 -> hyper_h) + ReLU
        Linear(hyper_h -> hyper_h) + ReLU
        Linear(hyper_h -> hyper_h) + ReLU
        Linear(hyper_h -> inr_param_count)

    The hypernetwork is the ONLY module with trainable parameters.
    """

    def __init__(
        self,
        inr_param_count: int,
        image_size: int = 784,
        hyper_h: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_size, hyper_h),
            nn.ReLU(),
            nn.Linear(hyper_h, hyper_h),
            nn.ReLU(),
            nn.Linear(hyper_h, hyper_h),
            nn.ReLU(),
            nn.Linear(hyper_h, inr_param_count),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # Override just the final output layer after the loop
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 784) flattened MNIST image, pixel values in [0, 1]

        Returns:
            (B, inr_param_count) flat weight vectors — one per image in the batch
        """
        return self.net(image)


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------


class HyperINR(nn.Module):
    """
    Full model: HyperNetwork + INRMLP.

    Only the HyperNetwork has trainable parameters.
    The INRMLP is a stateless functional template.

    Forward pass:
        1. image   -> hypernetwork -> flat_weights          (B, P)
        2. coords  + flat_weights  -> INR -> pixel_preds    (B, N, 1)
    """

    def __init__(self, h1: int = 20, h2: int = 20, h3: int = 20, omega_0: float = 20.0, hyper_h: int = 256):
        super().__init__()
        self.inr = INRMLP(h1=h1, h2=h2, h3=h3, omega_0=omega_0)
        inr_param_count = count_inr_params(h1, h2, h3)
        self.hypernet = HyperNetwork(inr_param_count=inr_param_count, hyper_h=hyper_h)

        print(f"INR parameter count   : {inr_param_count}")
        print(f"HyperNetwork out dim  : {inr_param_count}")
        print(f"HyperNetwork params   : {sum(p.numel() for p in self.hypernet.parameters()):,}")

    def forward(self, image, coords):
        flat_weights = self.hypernet(image)
        # print(f"flat_weights: min={flat_weights.min():.3f}, max={flat_weights.max():.3f}, mean={flat_weights.mean():.3f}")
        out = self.inr(coords, flat_weights)
        return out
