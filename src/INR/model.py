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
        """
        Args:
            x: (B, 2) normalized coordinates in [-1, 1]
        Returns:
            (B, 1) predicted pixel values in (0, 1)
        """
        return self.net(x)


# ----------------------------------------------------------------------
# Quick sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = INRMLP(h1=8, h2=16, h3=8)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    dummy = torch.rand(64, 2) * 2 - 1  # random coords in [-1, 1]
    out = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
