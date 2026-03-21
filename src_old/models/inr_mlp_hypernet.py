import torch.nn as nn
from src.models.Hypernetwork_MLP import HyperNetwork
from src.models.inr_siren import INRMLP
from src.utils.general_utils import count_inr_params


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
