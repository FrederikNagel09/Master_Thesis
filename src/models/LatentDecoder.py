import einops
import torch
from torch import nn

from src.models.trans_inr import instantiate_from_config, make_coord_grid, update_strategies


class MLPInr(nn.Module):
    """
    MLP-based INR hypernetwork. Replaces the tokenizer+transformer in TransInr
    with a flat MLP trunk + per-parameter heads that modulate a SIREN INR.

    Args:
        inr             : config dict for SIREN
        n_groups        : number of groups per INR parameter (must divide shape[1])
        data_shape      : (H, W) spatial resolution of the target output
        latent_dim      : channel depth of input feature map z
        latent_size     : (H', W') spatial size of input feature map z
        hidden_dim      : MLP trunk hidden dimension
        n_layers        : number of MLP trunk layers
        update_strategy : one of {"normalize", "scale", "identity"}
    """

    def __init__(
        self,
        inr: dict,
        n_groups: int,
        data_shape: tuple[int, int],
        latent_dim: int,
        latent_size: tuple[int, int],
        hidden_dim: int = 512,
        n_layers: int = 4,
        update_strategy: str = "scale",
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
        self.update_strategy = update_strategies[update_strategy]

        # --- Shared MLP trunk ---
        layers = [nn.Linear(flat_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)

        # --- Base INR params + per-parameter heads (mirrors TransInr) ---
        self.base_params = nn.ParameterDict()
        self.param_heads = nn.ModuleDict()
        self.wtoken_rng = {}

        for name, shape in self.inr.param_shapes.items():
            self.base_params[name] = nn.Parameter(self.inr.init_wb(shape, name=name))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0, f"n_groups={n_groups} must divide shape[1]={shape[1]} for layer {name}"
            # Each head maps hidden_dim → (shape[0]-1) * g, reshaped to (shape[0]-1, g)
            self.param_heads[name] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, (shape[0] - 1) * g),
            )
            self.wtoken_rng[name] = g

    def forward(self, data: torch.Tensor, coord: torch.Tensor | None = None, **kwargs) -> torch.Tensor:  # noqa: ARG002
        """
        Args:
            data  : (B, latent_dim, H', W') latent feature map
            coord : optional custom coordinate grid; uses shared_coord if None
        Returns:
            pred  : (B, C_out, H, W)
        """
        B = data.shape[0]  # noqa: N806

        # Flatten z and run through trunk
        h = self.trunk(data.reshape(B, -1))  # (B, hidden_dim)

        # Modulate base INR params via per-parameter heads
        params = {}
        for name, shape in self.inr.param_shapes.items():
            wb = einops.repeat(self.base_params[name], "n m -> b n m", b=B)
            w = wb[:, :-1, :]  # (B, shape[0]-1, shape[1])
            b = wb[:, -1:, :]  # (B, 1, shape[1])
            g = self.wtoken_rng[name]
            x = self.param_heads[name](h)  # (B, (shape[0]-1) * g)
            x = x.reshape(B, shape[0] - 1, g)  # (B, shape[0]-1, g)
            w = self.update_strategy(w, x)
            params[name] = torch.cat([w, b], dim=1)

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
