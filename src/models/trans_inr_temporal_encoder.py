"""
trans_inr_temporal_encoder.py
TransInr repurposed as a time-dependent weight encoder  F_phi(x, t).

Forward pass:
    x (B, C, H, W) or (B, data_dim)  +  t_norm (B, 1)
        → reshape x to spatial if needed
        → ImageTokenizer  →  patch tokens  (B, N_patch, dim)
        → prepend time token  →  (B, 1 + N_patch, dim)
        → Transformer enc+dec  →  trans_out  (B, N_w, dim)
        → modulate base_params
        → flatten  →  flat_weights  (B, weight_dim)

Public interface — satisfies NDMTemporalINR's F_phi contract:
    encoder = TransInrTemporalEncoder(...)
    flat_weights = encoder(x, t_norm)   # x: flat or spatial, t_norm: (B,1)
    encoder.weight_dim                  # int
    encoder.inr                         # SIREN (shared with NDM for decoding)
    encoder.inflate(flat_weights)       # (B, weight_dim) -> param dict
"""

import copy
import importlib
from typing import TYPE_CHECKING

import einops
import torch
import torch.nn as nn

from src.models.helper_modules import SinusoidalLearnableTimeEmbedding

if TYPE_CHECKING:
    from src.models.trans_inr_helpers import SIREN

# ---------------------------------------------------------------------------
# Config utilities
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
# Weight-update strategies
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
# TransInrTemporalEncoder
# ---------------------------------------------------------------------------


class TransInrTemporalEncoder(nn.Module):
    """
    Time-dependent TransInr weight encoder  F_phi(x, t).

    Identical to TransInrEncoder except a sinusoidal+learned time embedding
    is projected to `dim` and prepended as a single [TIME] token to the patch
    token sequence before the transformer encoder.  No positional embedding is
    added to the time token — its content already encodes its identity.

    Args
    ----
    tokenizer        : config dict for ImageTokenizer
    inr              : config dict for SIREN
    n_groups         : number of wtoken groups per INR parameter
    transformer      : config dict for Transformer (enc+dec)
    update_strategy  : one of {"normalize", "scale", "identity"}
    t_embed_dim      : dimension of the sinusoidal time embedding
    in_channels      : image channels (needed for flat→spatial reshape)
    img_size         : spatial size   (needed for flat→spatial reshape)
    """

    def __init__(
        self,
        tokenizer: dict,
        inr: dict,
        n_groups: int,
        transformer: dict,
        update_strategy: str = "normalize",
        t_embed_dim: int = 128,
        in_channels: int = 1,
        img_size: int = 28,
    ):
        super().__init__()

        dim = transformer["params"]["dim"]

        # ── Dataset shape (for flat→spatial reshape) ──────────────────────────
        self.in_channels = in_channels
        self.img_size = img_size

        # ── Sub-modules ───────────────────────────────────────────────────────
        self.tokenizer = instantiate_from_config(tokenizer, extra_args={"dim": dim})
        self.inr: SIREN = instantiate_from_config(inr)
        self.transformer = instantiate_from_config(transformer)

        # ── Time embedding ────────────────────────────────────────────────────
        # Sinusoidal + learned embedding → project to transformer dim
        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)
        self.time_proj = nn.Linear(t_embed_dim, dim)

        # ── Base INR parameters + wtoken machinery ────────────────────────────
        self.base_params = nn.ParameterDict()
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng: dict[str, tuple[int, int]] = {}

        n_wtokens = 0
        for name, shape in self.inr.param_shapes.items():
            self.base_params[name] = nn.Parameter(self.inr.init_wb(shape, name=name))
            g = min(n_groups, shape[1])
            while shape[1] % g != 0:
                g -= 1
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, shape[0] - 1),
            )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g

        self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))
        self.update_strategy = update_strategies[update_strategy]

        # ── weight_dim and shape bookkeeping ──────────────────────────────────
        self._weight_dim = sum(s[0] * s[1] for s in self.inr.param_shapes.values())
        self._param_names = list(self.inr.param_shapes.keys())
        self._param_shapes = dict(self.inr.param_shapes)

        nparams = (
            sum(p.numel() for p in self.transformer.parameters())
            + sum(p.numel() for p in self.tokenizer.parameters())
            + sum(p.numel() for p in self.base_params.values())
            + self.wtokens.numel()
            + sum(p.numel() for p in self.wtoken_postfc.parameters())
            + sum(p.numel() for p in self.time_embed.parameters())
            + sum(p.numel() for p in self.time_proj.parameters())
        )
        print(f"TransInrTemporalEncoder — total parameters: {nparams / 1e6:.3f}M")
        print(f"TransInrTemporalEncoder — weight_dim: {self._weight_dim}")

    # -------------------------------------------------------------------------
    # Public properties
    # -------------------------------------------------------------------------

    @property
    def weight_dim(self) -> int:
        return self._weight_dim

    # -------------------------------------------------------------------------
    # Flatten / inflate helpers  (identical to TransInrEncoder)
    # -------------------------------------------------------------------------

    def _flatten_params(self, param_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for name in self._param_names:
            wb = param_dict[name]  # (B, s0, s1)
            B = wb.shape[0]  # noqa: N806
            parts.append(wb.reshape(B, -1))
        return torch.cat(parts, dim=1)  # (B, weight_dim)

    def inflate(self, flat_weights: torch.Tensor) -> dict[str, torch.Tensor]:
        B = flat_weights.shape[0]  # noqa: N806
        param_dict = {}
        offset = 0
        for name in self._param_names:
            s0, s1 = self._param_shapes[name]
            n = s0 * s1
            chunk = flat_weights[:, offset : offset + n]
            param_dict[name] = chunk.reshape(B, s0, s1)
            offset += n
        return param_dict

    # -------------------------------------------------------------------------
    # Forward — (image, t) → flat weight vector
    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, C, H, W)  spatial  OR  (B, C*H*W)  flat  — reshaped internally
        t : (B, 1)         normalised time in [0, 1]

        Returns
        -------
        flat_weights : (B, weight_dim)
        """
        # ── 0. Flat → spatial if needed ───────────────────────────────────────
        if x.dim() == 2:
            x = x.view(x.shape[0], self.in_channels, self.img_size, self.img_size)

        B = x.shape[0]  # noqa: N806

        # ── 1. Build time token  (B, 1, dim) — no positional embedding ────────
        t_emb = self.time_embed(t)  # (B, t_embed_dim)
        t_tok = self.time_proj(t_emb).unsqueeze(1)  # (B, 1, dim)

        # ── 2. Tokenise image  →  (B, N_patch, dim) ───────────────────────────
        patch_tokens = self.tokenizer(x)  # (B, N_patch, dim)

        # ── 3. Prepend time token  →  (B, 1 + N_patch, dim) ──────────────────
        src = torch.cat([t_tok, patch_tokens], dim=1)  # (B, 1+N_patch, dim)

        # ── 4. Expand wtokens  →  (B, N_w, dim) ──────────────────────────────
        wtokens = einops.repeat(self.wtokens, "n d -> b n d", b=B)

        # ── 5. Transformer enc+dec ────────────────────────────────────────────
        cls_name = self.transformer.__class__.__name__
        if cls_name == "Transformer":
            trans_out = self.transformer(src=src, tgt=wtokens)
        elif cls_name == "TransformerEncoder":
            combined = torch.cat([src, wtokens], dim=1)
            full_out = self.transformer(combined)
            trans_out = full_out[:, -self.wtokens.shape[0] :, :]
        else:
            raise ValueError(f"Unsupported transformer class: {cls_name}")

        # ── 6. Modulate base INR parameters ───────────────────────────────────
        param_dict = {}
        for name, shape in self.inr.param_shapes.items():  # noqa: B007
            wb = einops.repeat(self.base_params[name], "n m -> b n m", b=B)
            w = wb[:, :-1, :]  # (B, s0-1, s1)
            b_row = wb[:, -1:, :]  # (B, 1,    s1)

            l, r = self.wtoken_rng[name]  # noqa: E741
            x_mod = self.wtoken_postfc[name](trans_out[:, l:r, :])
            x_mod = x_mod.transpose(-1, -2)  # (B, s0-1, g)
            w = self.update_strategy(w, x_mod)

            param_dict[name] = torch.cat([w, b_row], dim=1)  # (B, s0, s1)

        # ── 7. Flatten ────────────────────────────────────────────────────────
        return self._flatten_params(param_dict)  # (B, weight_dim)
