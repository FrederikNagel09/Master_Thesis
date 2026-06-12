import torch
from models.param_dit import ParamDiT


class WeightTransformation(ParamDiT):
    """
    F_phi(theta, t) for NDM-style transformation in weight space.

    Inherits ParamDiT architecture but applies an identity constraint:
        F_phi(theta, 0) = theta  exactly,  via  (1 - t) * theta + t * f_bar

    Args:
        param_shapes: dict mapping layer names to (out_dim, in_dim+1) shapes
        hidden_dim:   transformer hidden dimension
        depth:        number of transformer blocks
        num_heads:    number of attention heads
        mlp_ratio:    MLP expansion ratio
        dropout:      dropout rate
        time_dim:     timestep embedding dimension (defaults to hidden_dim)
    Returns:
        Transformed weight vector (B, modulation_dim), same shape as input
    """

    def forward(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            theta: (B, modulation_dim) weight vectors
            t:     (B, 1) normalised timestep in [0, 1]
        Returns:
            (B, modulation_dim) transformed weights
        """
        f_bar = super().forward(theta, t)  # ParamDiT forward
        return (1 - t) * theta + t * f_bar  # identity constraint
