import einops
import torch
from torch import nn

from src.models.trans_inr import instantiate_from_config, make_coord_grid


class MLPInr(nn.Module):
    """
    MLP-based INR hypernetwork. MLP trunk maps flattened latent → full INR weight vector,
    which is split and reshaped directly into INR parameters.
    Args:
        inr             : config dict for SIREN
        data_shape      : (H, W) spatial resolution of the target output
        latent_dim      : channel depth of input feature map z
        latent_size     : (H', W') spatial size of input feature map z
        hidden_dim      : MLP trunk hidden dimension
        n_layers        : number of MLP trunk layers
    """

    def __init__(
        self,
        inr: dict,
        data_shape: tuple[int, int],
        latent_dim: int,
        latent_size: tuple[int, int],
        hidden_dim: int = 512,
        n_layers: int = 4,
    ):
        super().__init__()
        latent_size = (latent_size, latent_size) if isinstance(latent_size, int) else latent_size
        flat_dim = latent_dim * latent_size[0] * latent_size[1]

        self.inr = instantiate_from_config(inr)
        self.register_buffer(
            "shared_coord",
            make_coord_grid(data_shape, (-1, 1)),
            persistent=False,
        )

        # Total number of INR parameters across all layers
        self.param_shapes = dict(self.inr.param_shapes)
        total_inr_params = sum(s[0] * s[1] for s in self.param_shapes.values())

        # Precompute split sizes for torch.split in forward
        self.split_sizes = [s[0] * s[1] for s in self.param_shapes.values()]

        # --- MLP trunk ---
        layers = [nn.Linear(flat_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)

        # Single projection head: hidden_dim → total INR params
        self.param_head = nn.Linear(hidden_dim, total_inr_params)

    def forward(self, data: torch.Tensor, coord: torch.Tensor | None = None, **kwargs) -> torch.Tensor:  # noqa: ARG002
        """
        Args:
            data  : (B, latent_dim, H', W') latent feature map
            coord : optional custom coordinate grid; uses shared_coord if None
        Returns:
            pred  : (B, C_out, H, W)
        """
        B = data.shape[0]  # noqa: N806

        # Flatten z, run trunk, project to full INR param vector
        h = self.trunk(data.reshape(B, -1))  # (B, hidden_dim)
        param_vec = self.param_head(h)  # (B, total_inr_params)

        # Split and reshape into per-layer (B, shape[0], shape[1]) tensors
        chunks = torch.split(param_vec, self.split_sizes, dim=-1)
        params = {name: chunk.reshape(B, *shape) for (name, shape), chunk in zip(self.param_shapes.items(), chunks, strict=False)}

        self.inr.set_params(params)

        # Query SIREN at pixel coordinates
        if coord is None:
            coord = self.shared_coord
        if coord.dim() == 3:
            coord = einops.repeat(coord, "h w d -> b h w d", b=B)

        pred = self.inr(coord)  # (B, H, W, C_out)
        if pred.dim() == 4:
            pred = pred.permute(0, 3, 1, 2).contiguous()
        return pred
