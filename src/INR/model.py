import torch
import torch.nn as nn


class INRMLP(nn.Module):
    """
    Three-layer MLP for implicit neural representation (INR).

    Architecture:
        Linear(2 -> h1) -> activation
        Linear(h1 -> h2) -> activation
        Linear(h2 -> h3) -> activation
        Linear(h3 -> 1)  -> Sigmoid

    Output is in (0, 1), matching pixel values normalized to [0, 1].

    Args:
        h1, h2, h3:  Hidden layer widths.
        activation:  Any nn.Module activation. Defaults to GELU, which tends
                     to work well for INRs. Sine activations (SIREN-style) are
                     another popular choice for this task.
    """

    def __init__(
        self,
        h1: int = 256,
        h2: int = 256,
        h3: int = 256,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        if activation is None:
            activation = nn.GELU()

        self.net = nn.Sequential(
            nn.Linear(2, h1),
            activation,
            nn.Linear(h1, h2),
            activation,
            nn.Linear(h2, h3),
            activation,
            nn.Linear(h3, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
