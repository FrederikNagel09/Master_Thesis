import torch
import torch.nn as nn

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
