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
import random
import sys
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Re-use helpers from trans_inr_helpers
# ---------------------------------------------------------------------------
import einops
import torch
import torch.nn as nn

sys.path.append(".")


from src.configs.general_config import GLOBAL_DEBUG_BOOL, probability_threshold
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
        probabilistic: bool = False,
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
        self._weight_dim = sum(shape[0] * shape[1] for shape in self.inr.param_shapes.values())

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
            assert shape[1] % g == 0, f"n_groups={n_groups} must divide shape[1]={shape[1]} for layer {name}"

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
        if x.dim() == 2:
            x = x.view(x.shape[0], self.in_channels, self.img_size, self.img_size)

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
        parts = [mod_dict[name].reshape(mod_dict[name].shape[0], -1) for name in self._param_names]
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
    def _compute_modulation_dicts(self, trans_out: torch.Tensor) -> tuple[dict, dict | None]:
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

    def encode_modulations(self, x: torch.Tensor, **kwargs) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Main encoder entrypoint. Returns compact modulation latents for Diffusion/VAE."""
        trans_out = self._run_transformer(x, **kwargs)
        mods, logvars = self._compute_modulation_dicts(trans_out)

        flat_mu = self.flatten_modulations(mods)
        if not self.probabilistic:
            return flat_mu

        flat_logvar = self.flatten_modulations(logvars)
        return flat_mu, flat_logvar

    # --- 4. DECODING PIPELINE ---
    def decode_modulations(self, flat_mods: torch.Tensor, return_dict: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
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
        parts = [param_dict[name].reshape(param_dict[name].shape[0], -1) for name in self._param_names]
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
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Acts as an end-to-end autoencoder mapping Image -> Modulations -> Raw Weights."""
        if not self.probabilistic:
            flat_mu_mods = self.encode_modulations(x, **kwargs)
            return flat_mu_mods

        # VAE Path
        flat_mu, flat_logvar = self.encode_modulations(x, **kwargs)

        return flat_mu, flat_logvar


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
        chunk_size: int = 128,
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
        self.time_mlp = nn.Sequential(nn.Linear(t_embed_dim, dim), nn.SiLU())

        # 3. Input Projection
        self.token_embed = nn.Linear(chunk_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4. Backbone: Reusing your TransformerEncoder
        self.transformer = TransformerEncoder(dim=dim, depth=depth, n_head=n_head, head_dim=head_dim, ff_dim=ff_dim, dropout=dropout)

        # 5. Output Head: Direct projection to noise
        self.noise_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, chunk_size))

        # Initialize head to zero or small values to help initial stability
        # nn.init.zeros_(self.noise_head[1].weight)
        # nn.init.zeros_(self.noise_head[1].bias)

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
        # === TIME SIGNAL DIAGNOSTIC ===
        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
            t_high = torch.ones_like(t)  # t=1.0 (999/999 = 1.0)
            t_low = torch.zeros_like(t)

            t_sin_high = self.time_embed(t_high)
            t_sin_low = self.time_embed(t_low)
            t_mlp_high = self.time_mlp(t_sin_high)
            t_mlp_low = self.time_mlp(t_sin_low)

            print(f"[DIAG] t shape: {t.shape}, values: {t.flatten()[:4]}")
            print(f"[DIAG] sinusoidal diff: {(t_sin_high - t_sin_low).abs().max():.6f}")
            print(f"[DIAG] after time_mlp: {(t_mlp_high - t_mlp_low).abs().max():.6f}")

        # ==============================
        # --- Step 2: Dense Conditioning (Option B) ---
        # Get time vector
        t_emb = self.time_mlp(self.time_embed(t))

        # Inject time and position into EVERY token
        x = x + self.pos_embed + t_emb.unsqueeze(1)

        # --- Step 3: Transformer Backbone ---
        x = self.transformer(x)  # (B, N, dim)

        # --- Step 4: Predict Noise ---
        out_tokens = self.noise_head(x)  # (B, N, chunk_size)

        # Flatten and unpad
        eps_hat = out_tokens.reshape(B, self.padded_dim)[:, : self.weight_dim]

        return eps_hat
