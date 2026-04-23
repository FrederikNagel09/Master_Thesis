"""
trans_inr_encoder.py
TransInr repurposed as a static weight encoder  W(x).

The full TransInr forward pass:
    image → ImageTokenizer → Transformer enc+dec → modulate base_params → SIREN(coord) → pixels

This module stops before the final INR query and instead returns a flat weight
vector that can be consumed by the NDM diffusion pipeline:
    image → ImageTokenizer → Transformer enc+dec → modulate base_params → flat_weights (B, weight_dim)

After diffusion, the flat vector can be inflated back into a param dict and
passed to the same SIREN via set_params() for decoding.

Public interface (compatible with NDMStaticINR's W encoder contract):
    encoder = TransInrEncoder(...)
    flat_weights = encoder(x)           # x: (B, C, H, W)
    encoder.weight_dim                  # int
    encoder.inr                         # SIREN instance  (shared with NDM for decoding)
    encoder.inflate(flat_weights)       # (B, weight_dim) -> param dict
"""

import copy
import importlib
import math
import sys

# ---------------------------------------------------------------------------
# Re-use helpers from trans_inr_helpers
# ---------------------------------------------------------------------------
import einops
import torch
import torch.nn as nn

sys.path.append(".")


from src.models.helper_modules import SinusoidalLearnableTimeEmbedding
from src.models.trans_inr_helpers import SIREN, TransformerEncoder

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


class INRModulator(nn.Module):
    """
    Modulates base INR parameters with (denoised) transformer output and
    exposes the SIREN for subsequent decoding.
    Receives flat trans_out (B, N_w * dim), inflates it, modulates base_params,
    and returns flat INR weights (B, weight_dim).
    Args
    ----
    inr              : config dict for SIREN
    n_groups         : number of wtoken groups per INR parameter
    dim              : transformer token dimension (must match encoder)
    update_strategy  : one of {"normalize", "scale", "identity"}
    """

    def __init__(
        self,
        inr: dict,
        n_groups: int,
        dim: int,
        update_strategy: str = "normalize",
    ):
        super().__init__()
        self.inr: SIREN = instantiate_from_config(inr)
        self._dim = dim

        self.base_params = nn.ParameterDict()
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng: dict[str, tuple[int, int]] = {}
        n_wtokens = 0
        for name, shape in self.inr.param_shapes.items():
            self.base_params[name] = nn.Parameter(self.inr.init_wb(shape, name=name))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0, f"n_groups={n_groups} must divide shape[1]={shape[1]} for layer {name}"
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, shape[0] - 1),
            )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g

        self._n_wtokens = n_wtokens
        self.update_strategy = update_strategies[update_strategy]

        self._weight_dim = sum(shape[0] * shape[1] for shape in self.inr.param_shapes.values())
        self._param_names: list[str] = list(self.inr.param_shapes.keys())
        self._param_shapes: dict[str, tuple[int, int]] = dict(self.inr.param_shapes)

        nparams = sum(p.numel() for p in self.base_params.values()) + sum(p.numel() for p in self.wtoken_postfc.parameters())
        print(f"INRModulator — total parameters: {nparams / 1e6:.3f}M")
        print(f"INRModulator — weight_dim: {self._weight_dim}")

    @property
    def weight_dim(self) -> int:
        """Flat INR weight dimension — output of modulation, input to SIREN."""
        return self._weight_dim

    def inflate_trans_out(self, flat_trans_out: torch.Tensor) -> torch.Tensor:
        """
        Reshape flat trans_out back to token sequence.
        Args
        ----
        flat_trans_out : (B, N_w * dim)
        Returns
        -------
        trans_out : (B, N_w, dim)
        """
        B = flat_trans_out.shape[0]  # noqa: N806
        return flat_trans_out.reshape(B, self._n_wtokens, self._dim)

    def inflate_weights(self, flat_weights: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Inflate flat INR weight vector back into a param dict.
        Args
        ----
        flat_weights : (B, weight_dim)
        Returns
        -------
        param_dict : {name: (B, shape[0], shape[1])}
        """
        B = flat_weights.shape[0]  # noqa: N806
        param_dict = {}
        offset = 0
        for name in self._param_names:
            s0, s1 = self._param_shapes[name]
            n = s0 * s1
            param_dict[name] = flat_weights[:, offset : offset + n].reshape(B, s0, s1)
            offset += n
        return param_dict

    def forward(self, flat_trans_out: torch.Tensor) -> torch.Tensor:
        """
        Modulate base INR params with denoised transformer output.
        Args
        ----
        flat_trans_out : (B, N_w * dim)  denoised, denormalized trans_out
        Returns
        -------
        flat_weights : (B, weight_dim)
        """
        trans_out = self.inflate_trans_out(flat_trans_out)  # (B, N_w, dim)
        B = trans_out.shape[0]  # noqa: N806

        param_dict = {}
        for name, shape in self.inr.param_shapes.items():  # noqa: B007
            wb = einops.repeat(self.base_params[name], "n m -> b n m", b=B)
            w = wb[:, :-1, :]  # (B, shape[0]-1, shape[1])
            b = wb[:, -1:, :]  # (B, 1, shape[1])

            l, r = self.wtoken_rng[name]  # noqa: E741
            x_mod = self.wtoken_postfc[name](trans_out[:, l:r, :])
            x_mod = x_mod.transpose(-1, -2)  # (B, shape[0]-1, g)

            w = self.update_strategy(w, x_mod)
            param_dict[name] = torch.cat([w, b], dim=1)  # (B, shape[0], shape[1])

        # Flatten all params into a single vector
        parts = [param_dict[name].reshape(B, -1) for name in self._param_names]
        return torch.cat(parts, dim=1)  # (B, weight_dim)


class TransInrEncoder(nn.Module):
    """
    TransInr repurposed as a static weight encoder W(x).
    Forward pass returns a flat transformer output vector (B, N_w * dim)
    intended for normalization → diffusion → INRModulator.
    Args
    ----
    tokenizer       : config dict for ImageTokenizer
    inr             : config dict for SIREN (used only for param_shapes + n_wtokens)
    n_groups        : number of wtoken groups per INR parameter
    transformer     : config dict for Transformer (enc+dec)
    in_channels     : image channels
    img_size        : spatial size of image
    """

    def __init__(
        self,
        tokenizer: dict,
        inr: dict,
        n_groups: int,
        transformer: dict,
        in_channels: int = 1,
        img_size: int = 28,
    ):
        super().__init__()
        dim = transformer["params"]["dim"]
        self.in_channels = in_channels
        self.img_size = img_size
        self._dim = dim

        self.tokenizer = instantiate_from_config(tokenizer, extra_args={"dim": dim})
        self.transformer = instantiate_from_config(transformer)

        # Build wtokens — one group per INR layer, no postfc here
        _inr_tmp = instantiate_from_config(inr)  # temp instance just for param_shapes
        n_wtokens = 0
        self.wtoken_rng: dict[str, tuple[int, int]] = {}
        for name, shape in _inr_tmp.param_shapes.items():
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0, f"n_groups={n_groups} must divide shape[1]={shape[1]} for layer {name}"
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
        self._n_wtokens = n_wtokens
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))

        # Flat dim of trans_out for the diffusion model to know
        self._trans_out_dim = n_wtokens * dim

        nparams = (
            sum(p.numel() for p in self.transformer.parameters())
            + sum(p.numel() for p in self.tokenizer.parameters())
            + self.wtokens.numel()
        )
        print(f"TransInrEncoder — total parameters: {nparams / 1e6:.3f}M")
        print(f"TransInrEncoder — trans_out_dim: {self._trans_out_dim}")

    @property
    def trans_out_dim(self) -> int:
        """Flat trans_out dimension — input dim for diffusion."""
        return self._trans_out_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode image to flat transformer output.
        Args
        ----
        x : (B, C, H, W) or (B, C*H*W)
        Returns
        -------
        flat_trans_out : (B, N_w * dim)
        """
        if x.dim() == 2:
            x = x.view(x.shape[0], self.in_channels, self.img_size, self.img_size)

        dtokens = self.tokenizer(x, **kwargs)  # (B, N_patch, dim)
        B = dtokens.shape[0]  # noqa: N806
        wtokens = einops.repeat(self.wtokens, "n d -> b n d", b=B)  # (B, N_w, dim)

        cls_name = self.transformer.__class__.__name__
        if cls_name == "Transformer":
            trans_out = self.transformer(src=dtokens, tgt=wtokens)  # (B, N_w, dim)
        elif cls_name == "TransformerEncoder":
            combined = torch.cat([dtokens, wtokens], dim=1)
            full_out = self.transformer(combined)
            trans_out = full_out[:, -self._n_wtokens :, :]
        else:
            raise ValueError(f"Unsupported transformer class: {cls_name}")

        return trans_out.reshape(B, -1)  # (B, N_w * dim)


class TransInrTemporalEncoder(nn.Module):
    """
    TransInr repurposed as a temporal weight encoder W(x, t).
    Forward pass returns a flat weight vector (B, weight_dim).
    The SIREN is exposed as self.inr so the NDM can use it for decoding.

    Args
    ----
    tokenizer        : config dict for ImageTokenizer
    inr              : config dict for SIREN
    n_groups         : number of wtoken groups per INR parameter
    transformer      : config dict for Transformer (enc+dec)
    update_strategy  : one of {"normalize", "scale", "identity"}
    time_freq_dim    : number of sinusoidal frequencies for time embedding
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
        time_freq_dim: int = 128,
    ):
        super().__init__()
        dim = transformer["params"]["dim"]
        self.in_channels = in_channels
        self.img_size = img_size

        # ── Time embedding ─────────────────────────────────────────────────────
        # Sinusoidal features → learned projection to dim
        self.time_freq_dim = time_freq_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_freq_dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # Fixed frequency bands — not a parameter
        freqs = torch.arange(1, time_freq_dim + 1, dtype=torch.float32)
        self.register_buffer("time_freqs", freqs)  # (time_freq_dim,)

        # ── Sub-modules ───────────────────────────────────────────────────────
        self.tokenizer = instantiate_from_config(tokenizer, extra_args={"dim": dim})
        self.inr: SIREN = instantiate_from_config(inr)
        self.transformer = instantiate_from_config(transformer)

        # ── Base INR parameters + wtoken machinery ────────────────────────────
        self.base_params = nn.ParameterDict()
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng: dict[str, tuple[int, int]] = {}
        n_wtokens = 0
        for name, shape in self.inr.param_shapes.items():
            self.base_params[name] = nn.Parameter(self.inr.init_wb(shape, name=name))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0, f"n_groups={n_groups} must divide shape[1]={shape[1]} for layer {name}"
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, shape[0] - 1),
            )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))
        self.update_strategy = update_strategies[update_strategy]

        self._weight_dim = sum(shape[0] * shape[1] for shape in self.inr.param_shapes.values())
        self._param_names: list[str] = list(self.inr.param_shapes.keys())
        self._param_shapes: dict[str, tuple[int, int]] = dict(self.inr.param_shapes)

        nparams = (
            sum(p.numel() for p in self.transformer.parameters())
            + sum(p.numel() for p in self.tokenizer.parameters())
            + sum(p.numel() for p in self.base_params.values())
            + self.wtokens.numel()
            + sum(p.numel() for p in self.wtoken_postfc.parameters())
            + sum(p.numel() for p in self.time_mlp.parameters())
        )
        print(f"TransInrTemporalEncoder — total parameters: {nparams / 1e6:.3f}M")
        print(f"TransInrTemporalEncoder — weight_dim: {self._weight_dim}")

    # -------------------------------------------------------------------------
    # Time embedding
    # -------------------------------------------------------------------------
    def _time_embedding(self, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal time embedding projected to transformer dim.
        Args:  t_norm : (B,) continuous time in [0, 1]
        Returns: (B, 1, dim) — ready to broadcast over token sequence
        """
        # (B, time_freq_dim) — sin and cos features over log-spaced frequencies
        angles = t_norm[:, None] * self.time_freqs[None, :] * torch.pi  # (B, F)
        t_emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, 2F)
        t_emb = self.time_mlp(t_emb)  # (B, dim)
        return t_emb.unsqueeze(1)  # (B, 1, dim)

    # -------------------------------------------------------------------------
    # Public properties
    # -------------------------------------------------------------------------
    @property
    def weight_dim(self) -> int:
        """Flat weight vector dimension — matches NDMStaticINR's expected weight_dim."""
        return self._weight_dim

    # -------------------------------------------------------------------------
    # Flatten / inflate helpers
    # -------------------------------------------------------------------------
    def _flatten_params(self, param_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Flatten an ordered param dict into a single vector per batch item.
        Args:    param_dict : {name: (B, shape[0], shape[1])}
        Returns: flat       : (B, weight_dim)
        """
        parts = []
        for name in self._param_names:
            wb = param_dict[name]
            B = wb.shape[0]  # noqa: N806
            parts.append(wb.reshape(B, -1))
        return torch.cat(parts, dim=1)

    def inflate(self, flat_weights: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Inflate a flat weight vector back into a param dict.
        Args:    flat_weights : (B, weight_dim)
        Returns: param_dict   : {name: (B, shape[0], shape[1])}
        """
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
    def forward(self, x: torch.Tensor, t_norm: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args
        ----
        x      : (B, C, H, W) or (B, C*H*W)  raw image tensor
        t_norm : (B,)  continuous timestep in [0, 1]
        Returns
        -------
        flat_weights : (B, weight_dim)
        """
        # ── 0. Flat → spatial if needed ───────────────────────────────────────
        if x.dim() == 2:
            x = x.view(x.shape[0], self.in_channels, self.img_size, self.img_size)

        # 1. Tokenise image → (B, N_patch, dim)
        dtokens = self.tokenizer(x, **kwargs)
        B = dtokens.shape[0]  # noqa: N806

        # 2. Time embedding → (B, 1, dim), inject into both token streams
        t_emb = self._time_embedding(t_norm)  # (B, 1, dim)
        wtokens = einops.repeat(self.wtokens, "n d -> b n d", b=B)
        dtokens = dtokens + t_emb  # broadcast over N_patch
        wtokens = wtokens + t_emb  # broadcast over N_w

        # 3. Transformer: image tokens → encoder, wtokens → decoder
        cls_name = self.transformer.__class__.__name__
        if cls_name == "Transformer":
            trans_out = self.transformer(src=dtokens, tgt=wtokens)
        elif cls_name == "TransformerEncoder":
            combined = torch.cat([dtokens, wtokens], dim=1)
            full_out = self.transformer(combined)
            trans_out = full_out[:, -self.wtokens.shape[0] :, :]
        else:
            raise ValueError(f"Unsupported transformer class: {cls_name}")

        # 4. Modulate base INR parameters with transformer output
        param_dict = {}
        for name, shape in self.inr.param_shapes.items():  # noqa: B007
            wb = einops.repeat(self.base_params[name], "n m -> b n m", b=B)
            w = wb[:, :-1, :]  # weight rows  (B, shape[0]-1, shape[1])
            b = wb[:, -1:, :]  # bias row     (B, 1,          shape[1])
            l, r = self.wtoken_rng[name]  # noqa: E741
            x_mod = self.wtoken_postfc[name](trans_out[:, l:r, :])
            x_mod = x_mod.transpose(-1, -2)
            w = self.update_strategy(w, x_mod)
            param_dict[name] = torch.cat([w, b], dim=1)

        # 5. Flatten to a single vector per batch item
        return self._flatten_params(param_dict)


class TransInrNoisePredictor(nn.Module):
    """ """

    def __init__(
        self,
        weight_dim: int,
        dim: int,
        depth: int,
        n_head: int,
        head_dim: int,
        ff_dim: int,
        chunk_size: int = 32,
        t_embed_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.weight_dim = weight_dim
        self.chunk_size = chunk_size
        self.dim = dim

        # 1. Chunking logic
        self.padded_dim = math.ceil(weight_dim / chunk_size) * chunk_size
        self.n_tokens = self.padded_dim // chunk_size

        # 2. Time Embedding (MLP for richer signal)
        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(t_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        # 3. Input Projection
        self.token_embed = nn.Linear(chunk_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4. Backbone: Reusing your TransformerEncoder
        self.transformer = TransformerEncoder(dim=dim, depth=depth, n_head=n_head, head_dim=head_dim, ff_dim=ff_dim, dropout=dropout)

        # 5. Output Head: Direct projection to noise
        self.noise_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, chunk_size))

        # Initialize head to zero or small values to help initial stability
        nn.init.zeros_(self.noise_head[1].weight)
        nn.init.zeros_(self.noise_head[1].bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (B, weight_dim)  noisy weight vector
            t : (B,)             timesteps
        """
        B = z.shape[0]  # noqa: N806

        # --- Step 1: Tokenize ---
        if self.padded_dim > self.weight_dim:
            pad = z.new_zeros(B, self.padded_dim - self.weight_dim)
            z_pad = torch.cat([z, pad], dim=-1)
        else:
            z_pad = z

        # Reshape to (B, N_tokens, Chunk_size)
        tokens = z_pad.view(B, self.n_tokens, self.chunk_size)
        x = self.token_embed(tokens)  # (B, N, dim)

        # --- Step 2: Dense Conditioning (Option B) ---
        # Get time vector
        t_emb = self.time_mlp(self.time_embed(t))  # (B, dim)

        # Inject time and position into EVERY token
        x = x + self.pos_embed + t_emb.unsqueeze(1)

        # --- Step 3: Transformer Backbone ---
        x = self.transformer(x)  # (B, N, dim)

        # --- Step 4: Predict Noise ---
        out_tokens = self.noise_head(x)  # (B, N, chunk_size)

        # Flatten and unpad
        eps_hat = out_tokens.reshape(B, self.padded_dim)[:, : self.weight_dim]

        return eps_hat
