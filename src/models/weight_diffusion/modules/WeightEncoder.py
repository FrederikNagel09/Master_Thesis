import copy
import importlib
import sys
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Re-use helpers from trans_inr_helpers
# ---------------------------------------------------------------------------
import einops
import torch
import torch.nn as nn

sys.path.append(".")


# ---------------------------------------------------------------------------
# Config utilities (copied from trans_inr.py to keep this file self-contained)
# ---------------------------------------------------------------------------


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, extra_args=None):
    if extra_args is not None:
        full_params = copy.deepcopy(config["params"])
        full_params.update(extra_args)
    else:
        full_params = config.get("params", dict())  # noqa: C408
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**full_params)


# ---------------------------------------------------------------------------
# Weight-update strategies (identical to trans_inr.py)
# ---------------------------------------------------------------------------


def normalize_weights(w, x):
    import torch.nn.functional as F  # noqa: N812

    return F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)


def scale_weights(w, x):
    return w * (1 + x.repeat(1, 1, w.shape[2] // x.shape[2]))


def identity_weights(w, x):  # noqa: ARG001
    return w


update_strategies = {
    "normalize": normalize_weights,
    "scale": scale_weights,
    "identity": identity_weights,
}


# ---------------------------------------------------------------------------
# TransInrEncoder
# ---------------------------------------------------------------------------


class TransInrEncoder(nn.Module):
    """
    TransInr weight encoder — updated for Late Modulation.

    Stage 1 (Autoencoder): forward() returns the final flat INR parameters.
    Stage 2 (Diffusion Target): encode_modulations() extracts the compact latent space.
    """

    def __init__(
        self,
        tokenizer: dict,
        inr: dict,
        n_groups: int,
        transformer: dict,
        update_strategy: str = "normalize",
        in_channels: int = 1,
        img_size: int = 28,
        probabilistic: bool = True,
    ):
        super().__init__()

        dim = transformer["params"]["dim"]
        self.in_channels = in_channels
        self.img_size = img_size
        self.probabilistic = probabilistic

        # Instantiate sub-modules
        self.tokenizer = instantiate_from_config(tokenizer, extra_args={"dim": dim})
        self.inr = instantiate_from_config(inr)
        self.transformer = instantiate_from_config(transformer)
        self.update_strategy = update_strategies[update_strategy]

        # Structure parameter definitions
        self._param_names = list(self.inr.param_shapes.keys())
        self._param_shapes = dict(self.inr.param_shapes)
        self._weight_dim = sum(
            shape[0] * shape[1] for shape in self.inr.param_shapes.values()
        )

        # Build base parameters and calculate compact modulation shapes
        self.base_params = nn.ParameterDict()
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = {}
        self.modulation_shapes = OrderedDict()

        n_wtokens = 0
        for name, shape in self.inr.param_shapes.items():
            # Base weights setup
            self.base_params[name] = nn.Parameter(self.inr.init_wb(shape, name=name))

            # Groups setup
            g = min(n_groups, shape[1])
            assert (
                shape[1] % g == 0
            ), f"n_groups={n_groups} must divide shape[1]={shape[1]} for layer {name}"

            weight_rows = shape[0] - 1
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, weight_rows),
            )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g

            # Store modulation shapes: (Rows modulated, Columns/Groups)
            self.modulation_shapes[name] = (weight_rows, g)

        self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))
        self._modulation_dim = sum(r * g for r, g in self.modulation_shapes.values())

        # Cleaner Probabilistic MLP targeting modulation space directly
        if self.probabilistic:
            self.logvar_mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, self._modulation_dim),
            )
            nn.init.zeros_(self.logvar_mlp[2].weight)
            nn.init.constant_(self.logvar_mlp[2].bias, -4.0)

    @property
    def weight_dim(self) -> int:
        return self._weight_dim

    @property
    def modulation_dim(self) -> int:
        return self._modulation_dim

    # --- 1. TOKENIZATION & TRANSFORMER CORE ---
    def _run_transformer(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Processes image through tokenizer and transformer."""
        print(f"Running transformer with input shape: {x.shape}")
        print(f"Image size: {self.img_size}, In channels: {self.in_channels}")
        if x.dim() == 2:
            x = x.view(x.shape[0], self.in_channels, self.img_size, self.img_size)

        print(f"Image shape after reshape: {x.shape}")

        dtokens = self.tokenizer(x, **kwargs)
        B = dtokens.shape[0]  # noqa: N806
        wtokens = einops.repeat(self.wtokens, "n d -> b n d", b=B)

        cls_name = self.transformer.__class__.__name__
        if cls_name == "Transformer":
            return self.transformer(src=dtokens, tgt=wtokens)
        elif cls_name == "TransformerEncoder":
            combined = torch.cat([dtokens, wtokens], dim=1)
            full_out = self.transformer(combined)
            return full_out[:, -self.wtokens.shape[0] :, :]
        else:
            raise ValueError(f"Unsupported transformer class: {cls_name}")

    # --- 2. MODULATION FLATTENING / UNFLATTENING ---
    def flatten_modulations(self, mod_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Flattens dict of layer modulations into a single vector (B, modulation_dim)."""
        parts = [
            mod_dict[name].reshape(mod_dict[name].shape[0], -1)
            for name in self._param_names
        ]
        return torch.cat(parts, dim=1)

    def unflatten_modulations(self, flat_mods: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inflates flat modulation vector back to a dictionary of layer shapes."""
        B = flat_mods.shape[0]  # noqa: N806
        mod_dict = {}
        offset = 0
        for name, shape in self.modulation_shapes.items():
            rows, g = shape
            numel = rows * g
            mod_dict[name] = flat_mods[:, offset : offset + numel].reshape(B, rows, g)
            offset += numel
        return mod_dict

    # --- 3. ENCODING PIPELINE (LATE MODULATION) ---
    def _compute_modulation_dicts(
        self, trans_out: torch.Tensor
    ) -> tuple[dict, dict | None]:
        """Extracts dictionaries of modulations (and optional logvars) from transformer tokens."""
        modulations = {}
        logvars = {} if self.probabilistic else None

        # Pull pooled features if mapping via MLP
        if self.probabilistic:
            pooled = trans_out.mean(dim=1)
            flat_logvars = self.logvar_mlp(pooled).clamp(-12.0, 4.0)
            logvars = self.unflatten_modulations(flat_logvars)

        for name in self._param_names:
            l, r = self.wtoken_rng[name]  # noqa: E741
            layer_tokens = trans_out[:, l:r, :]

            # (B, g, rows) -> transpose to standard (B, rows, g)
            x_mod = self.wtoken_postfc[name](layer_tokens).transpose(-1, -2)
            modulations[name] = x_mod

        return modulations, logvars

    def encode_modulations(
        self, x: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Main encoder entrypoint. Returns compact modulation latents for Diffusion/VAE."""
        trans_out = self._run_transformer(x, **kwargs)
        mods, logvars = self._compute_modulation_dicts(trans_out)

        flat_mu = self.flatten_modulations(mods)
        if not self.probabilistic:
            return flat_mu

        flat_logvar = self.flatten_modulations(logvars)
        return flat_mu, flat_logvar

    # --- 4. DECODING PIPELINE ---
    def decode_modulations(
        self, flat_mods: torch.Tensor, return_dict: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Decodes flat modulation vector by applying them to base weights."""
        B = flat_mods.shape[0]  # noqa: N806
        mod_dict = self.unflatten_modulations(flat_mods)
        param_dict = {}

        for name in self._param_names:
            # Replicate base weights across the current batch
            wb = einops.repeat(self.base_params[name], "n m -> b n m", b=B)
            w = wb[:, :-1, :]
            b = wb[:, -1:, :]

            # Apply update strategy using our unflattened modulations
            x_mod = mod_dict[name]
            w = self.update_strategy(w, x_mod)

            # Recombine weights + biases
            param_dict[name] = torch.cat([w, b], dim=1)

        if return_dict:
            return param_dict
        return self._flatten_params(param_dict)

    # --- 5. UTILITY PARAMETER FLATTENING ---
    def _flatten_params(self, param_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            param_dict[name].reshape(param_dict[name].shape[0], -1)
            for name in self._param_names
        ]
        return torch.cat(parts, dim=1)

    def inflate(self, flat_weights: torch.Tensor) -> dict[str, torch.Tensor]:
        B = flat_weights.shape[0]  # noqa: N806
        param_dict = {}
        offset = 0
        for name in self._param_names:
            s0, s1 = self._param_shapes[name]
            n = s0 * s1
            param_dict[name] = flat_weights[:, offset : offset + n].reshape(B, s0, s1)
            offset += n
        return param_dict

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    # --- 6. STANDARD FORWARD RULE (STAGE 1 AUTOENCODER) ---
    def forward(
        self, x: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Acts as an end-to-end autoencoder mapping Image -> Modulations -> Raw Weights."""
        if not self.probabilistic:
            flat_mu_mods = self.encode_modulations(x, **kwargs)
            return flat_mu_mods

        flat_mu, flat_logvar = self.encode_modulations(x, **kwargs)

        return flat_mu, flat_logvar
