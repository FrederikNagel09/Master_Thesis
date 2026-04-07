import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

sys.path.append(".")

from src.models.helper_modules import SinusoidalLearnableTimeEmbedding
from src.models.INR import INR, SirenINR  # noqa: F401


# =============================================================================
# Shared building block
# =============================================================================
class _ConvBlock(nn.Module):
    """Conv -> BatchNorm -> SiLU  (with optional stride for downsampling)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =============================================================================
# Noise Predictor Network  epsilon_theta(z_t, t)
# =============================================================================


class NoisePredictor(nn.Module):
    """
    MLP noise predictor  epsilon_theta(z_t, t).

    Operates entirely in *weight space* (the same space that F_phi maps into).
    For MNIST with the default INR this is `inr.num_weights` dimensional.

    Architecture
    ------------
    - Sinusoidal time embedding projected to `t_embed_dim`
    - Four residual blocks:  Linear -> LayerNorm -> SiLU -> Linear  (+ skip)
    - Time conditioning: add projected time embedding at the start of each block
    - Final linear readout with no activation (predicts raw noise)

    Parameters
    ----------
    weight_dim  : dimensionality of the weight vector (= inr.num_weights)
    hidden_dim  : width of every hidden layer            (default 512)
    n_blocks    : number of residual blocks              (default 4)
    t_embed_dim : dimensionality of the time embedding   (default 128)
    """

    def __init__(
        self,
        weight_dim: int,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        t_embed_dim: int = 128,
    ):
        super().__init__()

        self.weight_dim = weight_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks

        # --- Time embedding ---
        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)

        # Project time embedding to hidden_dim so we can add it inside each block
        self.time_proj = nn.Linear(t_embed_dim, hidden_dim)

        # --- Input projection: weight_dim -> hidden_dim ---
        self.input_proj = nn.Sequential(
            nn.Linear(weight_dim, hidden_dim),
            nn.SiLU(),
        )

        # --- Residual blocks ---
        # Each block: LayerNorm -> Linear -> SiLU -> Linear  (+skip from block input)
        # Time conditioning is added before the first activation.
        self.blocks = nn.ModuleList()
        self.t_projs = nn.ModuleList()  # one time projection per block
        for _ in range(n_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                    ]
                )
            )
            # separate time projections per block keep conditioning expressive
            self.t_projs.append(nn.Linear(t_embed_dim, hidden_dim))

        # --- Output projection: hidden_dim -> weight_dim ---
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, weight_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (batch, weight_dim)   noisy weight vector at time t
        t : (batch, 1)            normalised time in [0, 1]

        Returns
        -------
        eps_hat : (batch, weight_dim)  predicted noise
        """
        t_emb = self.time_embed(t)  # (batch, t_embed_dim)
        t_global = self.time_proj(t_emb)  # (batch, hidden_dim) — for residual conditioning

        h = self.input_proj(z)  # (batch, hidden_dim)
        h = h + t_global  # inject time before first block

        for (norm, lin1, lin2), t_proj in zip(self.blocks, self.t_projs):  # noqa: B905
            residual = h
            h = norm(h)
            h = lin1(h)
            h = h + t_proj(t_emb)  # time conditioning inside each block
            h = F.silu(h)
            h = lin2(h)
            h = h + residual  # skip connection

        return self.output_proj(h)  # (batch, weight_dim)


class TransformerNoisePredictor(nn.Module):
    """
    Transformer-based noise predictor  epsilon_theta(z_t, t).

    Tokenization
    ------------
    The flat weight vector of dimension `weight_dim` is split into
    fixed-size chunks of `chunk_size` dims each, giving a sequence of
    n_tokens = ceil(weight_dim / chunk_size) tokens.  If weight_dim is
    not divisible by chunk_size the vector is zero-padded to the next
    multiple before splitting, and the padding is discarded at the output.

    Time conditioning
    -----------------
    The time embedding is projected to d_model and prepended as a
    dedicated [TIME] token (index 0).  This lets every weight-space
    token attend to the global time signal via self-attention.

    Architecture
    ------------
    z (flat)  ->  pad & chunk  ->  linear per-token embed  ->  + pos embed
              ->  prepend [TIME] token
              ->  N x TransformerEncoderLayer
              ->  drop [TIME] token
              ->  linear per-token readout
              ->  reassemble  ->  drop padding  ->  weight vector (weight_dim)

    Parameters
    ----------
    weight_dim  : dimensionality of the weight vector
    chunk_size  : tokens are `chunk_size`-dimensional slices of the weight vec
    d_model     : transformer hidden dimension
    n_heads     : number of attention heads  (must divide d_model)
    n_layers    : number of transformer encoder layers
    d_ff        : feedforward dimension inside each transformer layer
    dropout     : dropout probability
    t_embed_dim : sinusoidal time embedding dimension
    """

    def __init__(
        self,
        weight_dim: int,
        chunk_size: int = 32,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        t_embed_dim: int = 128,
    ):
        super().__init__()
        self.weight_dim = weight_dim
        self.chunk_size = chunk_size
        self.d_model = d_model

        # Pad weight_dim to nearest multiple of chunk_size
        self.padded_dim = math.ceil(weight_dim / chunk_size) * chunk_size
        self.n_tokens = self.padded_dim // chunk_size

        # ── Time embedding ────────────────────────────────────────────────────
        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)
        self.time_proj = nn.Linear(t_embed_dim, d_model)

        # ── Per-token input projection: chunk_size -> d_model ─────────────────
        self.token_embed = nn.Linear(chunk_size, d_model)

        # ── Learnable positional embeddings (+ 1 for the TIME token) ──────────
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, S, d_model) convention throughout
            norm_first=True,  # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Per-token output projection: d_model -> chunk_size ────────────────
        self.token_readout = nn.Linear(d_model, chunk_size)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (batch, weight_dim)   noisy weight vector at time t
        t : (batch, 1)            normalised time in [0, 1]
        Returns
        -------
        eps_hat : (batch, weight_dim)  predicted noise
        """
        B = z.shape[0]  # noqa: N806

        # ── Pad and tokenize ──────────────────────────────────────────────────
        if self.padded_dim > self.weight_dim:
            pad = z.new_zeros(B, self.padded_dim - self.weight_dim)
            z_pad = torch.cat([z, pad], dim=-1)
        else:
            z_pad = z
        tokens = z_pad.view(B, self.n_tokens, self.chunk_size)  # (B, n_tokens, chunk_size)

        # ── Embed tokens ──────────────────────────────────────────────────────
        x = self.token_embed(tokens)  # (B, n_tokens, d_model)

        # ── Build TIME token and prepend ──────────────────────────────────────
        t_emb = self.time_embed(t)  # (B, t_embed_dim)
        t_tok = self.time_proj(t_emb).unsqueeze(1)  # (B, 1, d_model)
        x = torch.cat([t_tok, x], dim=1)  # (B, n_tokens+1, d_model)

        # ── Add positional embeddings ─────────────────────────────────────────
        x = x + self.pos_embed  # (B, n_tokens+1, d_model)

        # ── Transformer ───────────────────────────────────────────────────────
        x = self.transformer(x)  # (B, n_tokens+1, d_model)

        # ── Drop TIME token, project back to chunk_size ───────────────────────
        x = x[:, 1:, :]  # (B, n_tokens, d_model)
        x = self.token_readout(x)  # (B, n_tokens, chunk_size)

        # ── Reassemble and drop padding ───────────────────────────────────────
        eps_hat = x.reshape(B, self.padded_dim)[:, : self.weight_dim]
        return eps_hat  # (B, weight_dim)


# =============================================================================
# Data Transformation Network  F_phi(x, t) / W(x)
# =============================================================================


########## MLP Weight Encoders ##########
class MLPTemporalWeightEncoder(nn.Module):
    """
    MLP-based transformation network F_phi(x, t) for MNIST.

    Maps a flattened image x (784-dim) and normalised scalar time t to a
    *weight vector* in the INR parameter space (inr.num_weights - dim).

    Architecture
    ------------
    [x (784)] + [t_emb (t_embed_dim)]  ->  MLP  ->  weight vector (weight_dim)
    """

    def __init__(
        self,
        data_dim: int = 784,
        weight_dim: int = 501,  # must match inr.num_weights
        hidden_dims: list = None,  # noqa: RUF013
        t_embed_dim: int = 64,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)

        layers = []
        in_dim = data_dim + t_embed_dim
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.SiLU()]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, weight_dim))  # output = INR weight vector
        self.net = nn.Sequential(*layers)
        self.weight_dim = weight_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 784)   flattened MNIST image
        t : (batch, 1)     normalised time in [0, 1]

        Returns
        -------
        (batch, weight_dim)  weight vector in INR parameter space
        """
        t_emb = self.time_embed(t)  # (batch, t_embed_dim)
        xt = torch.cat([x, t_emb], dim=-1)
        return self.net(xt)


class MLPStaticWeightEncoder(nn.Module):
    """
    MLP-based transformation network W(x).
    Maps a flattened image x (784-dim) to a weight vector in the INR
    parameter space (weight_dim). No time conditioning.

    Architecture
    ------------
    [x (data_dim)]  ->  MLP  ->  weight vector (weight_dim)
    """

    def __init__(
        self,
        data_dim: int = 784,
        weight_dim: int = 501,
        hidden_dims: list = None,  # noqa: RUF013
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        layers = []
        in_dim = data_dim
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.SiLU()]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, weight_dim))
        self.net = nn.Sequential(*layers)
        self.weight_dim = weight_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, data_dim)   flattened input image
        Returns
        -------
        (batch, weight_dim)  weight vector in INR parameter space
        """
        return self.net(x)


########## CNN Weight Encoders ##########
class CNNTemporalWeightEncoder(nn.Module):
    """
    CNN-based time-dependent weight encoder  F_phi(x, t).

    Pipeline
    --------
    x (flat)  ->  reshape (B, C, H, W)
              ->  CNN backbone (conv blocks + global avg pool)
              ->  concat time embedding
              ->  linear projection  ->  weight vector (weight_dim)

    The spatial structure of the image is exploited by the CNN before
    being combined with the time signal and projected to weight space.

    Parameters
    ----------
    data_dim    : flat image dimension  (e.g. 784 for MNIST)
    img_size    : spatial size          (e.g. 28 for MNIST)
    channels    : image channels        (1 for grayscale, 3 for RGB)
    weight_dim  : output dimension      (= inr.num_weights)
    base_ch     : base channel width for CNN  (doubled each block)
    n_blocks    : number of conv blocks
    t_embed_dim : time embedding dimension
    """

    def __init__(
        self,
        data_dim: int = 784,
        img_size: int = 28,
        channels: int = 1,
        weight_dim: int = 501,
        base_ch: int = 32,
        n_blocks: int = 4,
        t_embed_dim: int = 64,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.img_size = img_size
        self.channels = channels

        # ── Time embedding ────────────────────────────────────────────────────
        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)

        # ── CNN backbone ──────────────────────────────────────────────────────
        # Each block doubles channels and halves spatial dims via stride=2
        # except the first block which just lifts channel count
        cnn_layers = []
        in_ch = channels
        out_ch = base_ch
        for i in range(n_blocks):
            stride = 1 if i == 0 else 2
            cnn_layers.append(_ConvBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)  # cap at 512 channels
        self.cnn = nn.Sequential(*cnn_layers)

        # Global average pooling -> flat feature vector of size in_ch
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Projection: CNN features + time -> weight_dim ─────────────────────
        self.proj = nn.Sequential(
            nn.Linear(in_ch + t_embed_dim, in_ch * 2),
            nn.SiLU(),
            nn.Linear(in_ch * 2, weight_dim),
        )
        self.weight_dim = weight_dim
        self.out_norm = nn.LayerNorm(weight_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, data_dim)   flattened image
        t : (batch, 1)          normalised time in [0, 1]
        Returns
        -------
        (batch, weight_dim)
        """
        # Reshape flat image -> spatial
        x2d = x.view(-1, self.channels, self.img_size, self.img_size)

        # CNN feature extraction
        feat = self.cnn(x2d)  # (B, C_out, H', W')
        feat = self.gap(feat).flatten(1)  # (B, C_out)

        # Time embedding
        t_emb = self.time_embed(t)  # (B, t_embed_dim)

        # Combine and project
        out = self.proj(torch.cat([feat, t_emb], dim=-1))
        return self.out_norm(out)  # (B, weight_dim)


class CNNStaticWeightEncoder(nn.Module):
    """
    CNN-based time-independent weight encoder  W(x).

    Same CNN backbone as CNNTemporalWeightEncoder but without any
    time conditioning — a clean drop-in for StaticWeightEncoder.

    Parameters
    ----------
    data_dim   : flat image dimension  (e.g. 784 for MNIST)
    img_size   : spatial size          (e.g. 28 for MNIST)
    channels   : image channels        (1 for grayscale, 3 for RGB)
    weight_dim : output dimension      (= inr.num_weights)
    base_ch    : base channel width for CNN  (doubled each block)
    n_blocks   : number of conv blocks
    """

    def __init__(
        self,
        data_dim: int = 784,
        img_size: int = 28,
        channels: int = 1,
        weight_dim: int = 501,
        base_ch: int = 32,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.img_size = img_size
        self.channels = channels

        # ── CNN backbone ──────────────────────────────────────────────────────
        cnn_layers = []
        in_ch = channels
        out_ch = base_ch
        for i in range(n_blocks):
            stride = 1 if i == 0 else 2
            cnn_layers.append(_ConvBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
        self.cnn = nn.Sequential(*cnn_layers)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Projection: CNN features -> weight_dim ────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(in_ch, in_ch * 2),
            nn.SiLU(),
            nn.Linear(in_ch * 2, weight_dim),
        )
        self.weight_dim = weight_dim
        self.out_norm = nn.LayerNorm(weight_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, data_dim)   flattened image
        Returns
        -------
        (batch, weight_dim)
        """
        x2d = x.view(-1, self.channels, self.img_size, self.img_size)
        feat = self.cnn(x2d)
        feat = self.gap(feat).flatten(1)  # (B, C_out)
        out = self.proj(feat)
        return self.out_norm(out)  # (B, weight_dim)


########## Transformer Weight Encoders ##########
class TransformerTemporalWeightEncoder(nn.Module):
    """
    Transformer-based time-dependent weight encoder  F_phi(x, t).

    Pipeline
    --------
    x (flat)  ->  reshape (B, C, H, W)
              ->  patch embedding  (B, num_patches, embed_dim)
              ->  + positional embedding
              ->  Transformer encoder blocks
              ->  CLS token output
              ->  concat time embedding
              ->  linear projection  ->  weight vector (weight_dim)

    Parameters
    ----------
    data_dim    : flat image dimension  (e.g. 784 for MNIST)
    img_size    : spatial size          (e.g. 28 for MNIST)
    channels    : image channels        (1 for grayscale, 3 for RGB)
    weight_dim  : output dimension      (= inr.num_weights)
    patch_size  : size of each square patch  (img_size must be divisible)
    embed_dim   : transformer embedding dimension
    n_blocks    : number of transformer encoder layers
    n_heads     : number of attention heads
    mlp_ratio   : MLP hidden dim multiplier inside transformer
    t_embed_dim : time embedding dimension
    dropout     : dropout rate
    """

    def __init__(
        self,
        data_dim: int = 784,
        img_size: int = 28,
        channels: int = 1,
        weight_dim: int = 501,
        patch_size: int = 4,
        embed_dim: int = 128,
        n_blocks: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        t_embed_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.data_dim = data_dim
        self.img_size = img_size
        self.channels = channels

        num_patches = (img_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        # ── Patch embedding ───────────────────────────────────────────────────
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # Learnable CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.norm = nn.LayerNorm(embed_dim)

        # ── Time embedding ────────────────────────────────────────────────────
        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)

        # ── Projection: CLS features + time -> weight_dim ────────────────────
        self.proj = nn.Sequential(
            nn.Linear(embed_dim + t_embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, weight_dim),
        )
        self.weight_dim = weight_dim
        self.out_norm = nn.LayerNorm(weight_dim)

    def _patchify(self, x2d: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_patches, patch_dim)"""
        B, C, H, W = x2d.shape  # noqa: N806
        p = self.patch_size
        x2d = x2d.reshape(B, C, H // p, p, W // p, p)
        x2d = x2d.permute(0, 2, 4, 1, 3, 5)  # (B, H/p, W/p, C, p, p)
        return x2d.flatten(1, 2).flatten(2)  # (B, num_patches, patch_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, data_dim)   flattened image
        t : (batch, 1)          normalised time in [0, 1]
        Returns
        -------
        (batch, weight_dim)
        """
        B = x.size(0)  # noqa: N806
        x2d = x.view(B, self.channels, self.img_size, self.img_size)

        # Patchify and embed
        patches = self._patchify(x2d)  # (B, N, patch_dim)
        tokens = self.patch_embed(patches)  # (B, N, embed_dim)

        # Prepend CLS token and add positional embeddings
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N+1, embed_dim)
        tokens = tokens + self.pos_embed

        # Transformer
        tokens = self.transformer(tokens)  # (B, N+1, embed_dim)
        cls_out = self.norm(tokens[:, 0])  # (B, embed_dim)

        # Time embedding
        t_emb = self.time_embed(t)  # (B, t_embed_dim)

        # Combine and project
        out = self.proj(torch.cat([cls_out, t_emb], dim=-1))
        return self.out_norm(out)  # (B, weight_dim)


class TransformerStaticWeightEncoder(nn.Module):
    """
    Transformer-based time-independent weight encoder  W(x).

    Same ViT-style backbone as TransformerTemporalWeightEncoder but
    without any time conditioning — a clean drop-in for
    CNNStaticWeightEncoder.

    Parameters
    ----------
    data_dim   : flat image dimension  (e.g. 784 for MNIST)
    img_size   : spatial size          (e.g. 28 for MNIST)
    channels   : image channels        (1 for grayscale, 3 for RGB)
    weight_dim : output dimension      (= inr.num_weights)
    patch_size : size of each square patch  (img_size must be divisible)
    embed_dim  : transformer embedding dimension
    n_blocks   : number of transformer encoder layers
    n_heads    : number of attention heads
    mlp_ratio  : MLP hidden dim multiplier inside transformer
    dropout    : dropout rate
    """

    def __init__(
        self,
        data_dim: int = 784,
        img_size: int = 28,
        channels: int = 1,
        weight_dim: int = 501,
        patch_size: int = 4,
        embed_dim: int = 128,
        n_blocks: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.data_dim = data_dim
        self.img_size = img_size
        self.channels = channels

        num_patches = (img_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        # ── Patch embedding ───────────────────────────────────────────────────
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.norm = nn.LayerNorm(embed_dim)

        # ── Projection: CLS features -> weight_dim ────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, weight_dim),
        )
        self.weight_dim = weight_dim
        self.out_norm = nn.LayerNorm(weight_dim)

    def _patchify(self, x2d: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_patches, patch_dim)"""
        B, C, H, W = x2d.shape  # noqa: N806
        p = self.patch_size
        x2d = x2d.reshape(B, C, H // p, p, W // p, p)
        x2d = x2d.permute(0, 2, 4, 1, 3, 5)
        return x2d.flatten(1, 2).flatten(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, data_dim)   flattened image
        Returns
        -------
        (batch, weight_dim)
        """
        B = x.size(0)  # noqa: N806
        x2d = x.view(B, self.channels, self.img_size, self.img_size)

        patches = self._patchify(x2d)
        tokens = self.patch_embed(patches)  # (B, N, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed

        tokens = self.transformer(tokens)
        cls_out = self.norm(tokens[:, 0])  # (B, embed_dim)

        out = self.proj(cls_out)
        return self.out_norm(out)  # (B, weight_dim)


# =============================================================================
# Base Neural Diffusion Model  — shared logic
# =============================================================================


class NeuralDiffusionModelINR(nn.Module):
    """
    Base class for Neural Diffusion Models with INR-based reconstruction.
    Contains all shared logic: noise schedule, INR decoding, sigma_tilde,
    modulation, and the ELBO structure.

    Subclasses must implement:
        _sample_zt(x, t_idx, t_norm)  ->  (z_t, epsilon, Wx)
        _l_diff(x, z_t, t_idx, t_norm, Wx)  ->  (batch,)
        _l_prior(x)  ->  (batch,)
        _l_rec(x)    ->  (batch,)
        sample_weight(n_samples)  ->  (n_samples, weight_dim)
    """

    def __init__(
        self,
        network: nn.Module,
        inr: INR,
        beta_1: float = 1e-4,
        beta_T: float = 2e-2,  # noqa: N803
        T: int = 100,  # noqa: N803
        sigma_tilde_factor: float = 1.0,
        data_dim: int = 784,
        img_size: int = 28,
        use_modulation: bool = False,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.img_size = img_size
        self.network = network
        self.inr = inr
        self.use_modulation = use_modulation

        # ── Learnable base weight vector ──────────────────────────────────────
        # Initialise eagerly so load_state_dict can always find the key
        if use_modulation:
            self._theta_b = nn.Parameter(torch.empty(1, inr.num_weights))
            nn.init.normal_(self._theta_b, std=0.1)
        else:
            self._theta_b = None

        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.sigma_tilde_factor = sigma_tilde_factor

        # --- Noise schedule ---
        beta = torch.linspace(beta_1, beta_T, T)
        alpha = 1.0 - beta
        alpha_cumprod = alpha.cumprod(dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", alpha_cumprod.sqrt())
        self.register_buffer("sigma_sq", 1.0 - alpha_cumprod)
        self.register_buffer("sigma", (1.0 - alpha_cumprod).sqrt())

        # --- Pre-build pixel coordinate grid for MNIST (28x28) ---
        xs = torch.linspace(-1, 1, img_size)
        ys = torch.linspace(-1, 1, img_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (784, 2)
        self.register_buffer("coords", coords.unsqueeze(0))  # (1, 784, 2)

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------
    def _sigma_tilde_sq(self, s_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        sigma_s_sq = self.sigma_sq[s_idx]
        sigma_t_sq = self.sigma_sq[t_idx]
        alpha_t_sq = self.alpha_cumprod[t_idx]
        alpha_s_sq = self.alpha_cumprod[s_idx]
        base = (sigma_t_sq - alpha_t_sq / alpha_s_sq * sigma_s_sq) * sigma_s_sq / sigma_t_sq
        return self.sigma_tilde_factor * base

    def _modulate(self, theta: torch.Tensor) -> torch.Tensor:
        if not self.use_modulation:
            return theta
        return (1.0 + theta) * self._theta_b

    def _inr_decode(self, flat_weights: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        """
        Decode a batch of flat weight vectors to pixel images.
        Parameters
        ----------
        flat_weights : (batch, num_weights)
        Returns
        -------
        pixels : (batch, 784)
        """
        batch = flat_weights.shape[0]
        if self.use_modulation:
            flat_weights = self._modulate(flat_weights)
        if coords is None:
            coords = self.coords
        coords = coords.expand(batch, -1, -1)
        pixels = self.inr(coords, flat_weights)
        return pixels.squeeze(-1)

    # -------------------------------------------------------------------------
    # Abstract interface — subclasses must override these
    # -------------------------------------------------------------------------
    def _sample_zt(self, x, t_idx, t_norm):
        raise NotImplementedError

    def _l_diff(self, x, z_t, t_idx, t_norm, Wx):  # noqa: N803
        raise NotImplementedError

    def _l_prior(self, x):
        raise NotImplementedError

    def _l_rec(self, x):
        raise NotImplementedError

    def sample_weight(self, n_samples: int = 1):
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Shared ELBO structure
    # -------------------------------------------------------------------------
    def negative_elbo(self, x: torch.Tensor):
        """
        Estimates the negative ELBO:
            L = E[ l_diff ] + prior_mask * l_prior + l_rec
        Parameters
        ----------
        x : (batch, data_dim)
        Returns
        -------
        (scalar mean loss, l_diff mean, l_prior mean, l_rec mean)
        """
        batch_size = x.shape[0]

        # Sample random time step  t ~ Uniform{1, ..., T}
        t_idx = torch.randint(1, self.T + 1, (batch_size,), device=x.device) - 1
        t_norm = t_idx.float() / (self.T - 1)

        # Forward: z_t ~ q(z_t | x)
        z_t, _, Wx = self._sample_zt(x, t_idx, t_norm.unsqueeze(1))  # noqa: N806

        # Three loss terms
        l_diff = self._l_diff(x, z_t, t_idx, t_norm, Wx)  # (batch,)
        l_prior = self._l_prior(x)  # (batch,)
        l_rec = self._l_rec(x)  # (batch,)

        prior_mask = (t_idx == self.T - 1).float()
        elbo = l_diff + prior_mask * l_prior + l_rec

        return elbo.mean(), l_diff.mean(), l_prior.mean(), l_rec.mean()

    def loss(self, x: torch.Tensor):
        return self.negative_elbo(x)

    # -------------------------------------------------------------------------
    # Shared decode / sample convenience wrappers
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def decode_weights(self, weights: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        return self._inr_decode(weights, coords)

    @torch.no_grad()
    def sample(self, n_samples: int = 1, coords: torch.Tensor | None = None) -> torch.Tensor:
        weights = self.sample_weight(n_samples)
        return self.decode_weights(weights, coords)


# =============================================================================
# Subclass 1 — Time-dependent  F_phi(x, t)
# =============================================================================
class NDMTemporalINR(NeuralDiffusionModelINR):
    """
    NDM with a time-dependent data transformation F_phi(x, t).
    Forward process:
        z_t = alpha_t * F_phi(x, t) + sigma_t * eps,   eps ~ N(0, I)
    """

    def __init__(
        self,
        network: nn.Module,
        F_phi: MLPTemporalWeightEncoder | CNNTemporalWeightEncoder,  # noqa: N803
        inr: INR,
        **kwargs,
    ):
        super().__init__(network=network, inr=inr, **kwargs)
        self.F_phi = F_phi

    # -------------------------------------------------------------------------
    # Forward process
    # -------------------------------------------------------------------------
    def _sample_zt(self, x, t_idx, t_norm):
        """Returns z_t, epsilon, and F_phi(x, t)."""
        Fx = self.F_phi(x, t_norm)  # (batch, weight_dim)  # noqa: N806
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)
        epsilon = torch.randn_like(Fx)
        z_t = alpha_t * Fx + sigma_t * epsilon
        return z_t, epsilon, Fx

    # -------------------------------------------------------------------------
    # Loss terms
    # -------------------------------------------------------------------------
    def _l_diff(self, x, z_t, t_idx, t_norm, Fx_t):  # noqa: N803
        eps_hat = self.network(z_t, t_norm.unsqueeze(1))  # (batch, weight_dim)
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)

        # Recover predicted clean weight vector from predicted noise
        x_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

        s_idx = (t_idx - 1).clamp(min=0)
        s_norm = s_idx.float() / (self.T - 1)

        # Decode predicted weights to pixel space, then re-encode with F_phi
        x_hat_pixels = self._inr_decode(x_hat)
        Fx_hat_t = self.F_phi(x_hat_pixels, t_norm.unsqueeze(1))  # noqa: N806
        Fx_hat_s = self.F_phi(x_hat_pixels, s_norm.unsqueeze(1))  # noqa: N806
        Fx_s = self.F_phi(x, s_norm.unsqueeze(1))  # noqa: N806

        alpha_s = self.sqrt_alpha_cumprod[s_idx].unsqueeze(1)
        sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx).unsqueeze(1)
        coeff = (self.sigma_sq[s_idx].unsqueeze(1) - sigma_tilde_sq).clamp(min=0).sqrt() / self.sigma[t_idx].unsqueeze(1).clamp(min=1e-6)

        diff = alpha_s * (Fx_s - Fx_hat_s) + coeff * alpha_t * (Fx_hat_t - Fx_t)
        l_diff = (diff**2).sum(dim=-1) / (2.0 * sigma_tilde_sq.squeeze(1).clamp(min=1e-8))
        return l_diff

    def _l_prior(self, x: torch.Tensor) -> torch.Tensor:
        """Closed-form KL  N(alpha_T * F(x,T), sigma_T^2 I) || N(0,I)."""
        T_idx = self.T - 1  # noqa: N806
        t_norm_T = torch.ones(x.shape[0], 1, device=x.device)  # noqa: N806
        Fx_T = self.F_phi(x, t_norm_T)  # noqa: N806
        sigma_T_sq = self.sigma_sq[T_idx]  # noqa: N806
        alpha_T_sq = self.alpha_cumprod[T_idx]  # noqa: N806
        d = Fx_T.shape[-1]
        kl = 0.5 * (d * (sigma_T_sq - torch.log(sigma_T_sq) - 1.0) + alpha_T_sq * (Fx_T**2).sum(dim=-1))
        return kl

    def _l_rec(self, x: torch.Tensor) -> torch.Tensor:
        """INR-based reconstruction loss at t = 0."""
        t0_norm = torch.zeros(x.shape[0], 1, device=x.device)
        weights = self.F_phi(x, t0_norm)
        x_recon = self._inr_decode(weights)
        return 0.5 * ((x - x_recon) ** 2).sum(dim=-1)

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def sample_weight(self, n_samples: int = 1) -> torch.Tensor:
        weight_dim = self.F_phi.weight_dim
        device = self.sqrt_alpha_cumprod.device
        theta_t = torch.randn(n_samples, weight_dim, device=device)

        for t in tqdm(range(self.T - 1, -1, -1), desc="NDM (Temporal) Sampling", total=self.T):
            t_idx = torch.full((n_samples,), t, dtype=torch.long, device=device)
            t_norm = torch.full((n_samples, 1), t / max(self.T - 1, 1), device=device)

            eps_hat = self.network(theta_t, t_norm)
            alpha_t = self.sqrt_alpha_cumprod[t].unsqueeze(0)
            sigma_t = self.sigma[t].unsqueeze(0)
            theta_t_hat = (theta_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

            if t == 0:
                return theta_t_hat

            s = t - 1
            s_idx = torch.full((n_samples,), s, dtype=torch.long, device=device)
            s_norm = torch.full((n_samples, 1), s / max(self.T - 1, 1), device=device)

            x_hat_pixels = self._inr_decode(theta_t_hat)
            Fx_hat_t = self.F_phi(x_hat_pixels, t_norm)  # noqa: N806
            Fx_hat_s = self.F_phi(x_hat_pixels, s_norm)  # noqa: N806

            alpha_s = self.sqrt_alpha_cumprod[s].view(1, 1)
            sigma_s_sq = self.sigma_sq[s].view(1, 1)
            sigma_t_val = self.sigma[t].view(1, 1)
            alpha_t_val = self.sqrt_alpha_cumprod[t].view(1, 1)
            sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx)[0].view(1, 1)
            coeff = (sigma_s_sq - sigma_tilde_sq).clamp(min=0).sqrt() / sigma_t_val.clamp(min=1e-6)

            mu = alpha_s * Fx_hat_s + coeff * (theta_t - alpha_t_val * Fx_hat_t)
            noise = torch.randn_like(theta_t) if sigma_tilde_sq.item() > 0 else torch.zeros_like(theta_t)
            theta_t = mu + sigma_tilde_sq.clamp(min=0).sqrt() * noise

        return theta_t_hat  # safety fallback


# =============================================================================
# Subclass 2 — Time-independent  W(x)
# =============================================================================
class NDMStaticINR(NeuralDiffusionModelINR):
    """
    NDM with a time-independent data transformation W(x).
    Forward process:
        z_t = alpha_t * W(x) + sigma_t * eps,   eps ~ N(0, I)

    The diffusion loss simplifies to a single squared error in W-space:
        L_diff = 1 / (2 * sigma_tilde_sq) * (alpha_s - B * alpha_t)^2
                 * || W(x) - W(x_hat) ||^2
    where B = sqrt(sigma_s^2 - sigma_tilde^2) / sigma_t.
    """

    def __init__(
        self,
        network: nn.Module,
        W: MLPStaticWeightEncoder | CNNStaticWeightEncoder,  # noqa: N803
        inr: INR,
        **kwargs,
    ):
        super().__init__(network=network, inr=inr, **kwargs)
        self.W = W

    # -------------------------------------------------------------------------
    # Forward process
    # -------------------------------------------------------------------------
    def _sample_zt(self, x, t_idx, t_norm):  # noqa: ARG002
        """Returns z_t, epsilon, and W(x). t_norm unused but kept for API consistency."""
        Wx = self.W(x)  # (batch, weight_dim)  # noqa: N806
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)
        epsilon = torch.randn_like(Wx)
        z_t = alpha_t * Wx + sigma_t * epsilon
        return z_t, epsilon, Wx

    # -------------------------------------------------------------------------
    # Loss terms
    # -------------------------------------------------------------------------
    def _l_diff(self, x, z_t, t_idx, t_norm, Wx):  # noqa: ARG002, N803
        """
        Simplified L_diff for time-independent W(x).

        Because W has no time dependence, W(x,s) = W(x,t) = W(x), so the
        two terms in the general loss factor out into a single squared error:

            diff = (alpha_s - B * alpha_t) * (W(x) - W(x_hat))

        where B = sqrt(sigma_s^2 - sigma_tilde^2) / sigma_t.
        """
        eps_hat = self.network(z_t, t_norm.unsqueeze(1))  # (batch, weight_dim)
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)

        # Recover predicted clean weight vector
        x_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

        s_idx = (t_idx - 1).clamp(min=0)

        # Decode predicted weights to pixel space, then re-encode with W
        x_hat_pixels = self._inr_decode(x_hat)
        Wx_hat = self.W(x_hat_pixels)  # (batch, weight_dim)  # noqa: N806

        alpha_s = self.sqrt_alpha_cumprod[s_idx].unsqueeze(1)
        sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx).unsqueeze(1)
        B = (  # noqa: N806
            (self.sigma_sq[s_idx].unsqueeze(1) - sigma_tilde_sq).clamp(min=0).sqrt() / self.sigma[t_idx].unsqueeze(1).clamp(min=1e-6)
        )

        # Scalar coefficient — factored out because W has no time dependence
        coeff = alpha_s - B * alpha_t  # (batch, 1)
        diff = coeff * (Wx - Wx_hat)  # (batch, weight_dim)

        l_diff = (diff**2).sum(dim=-1) / (2.0 * sigma_tilde_sq.squeeze(1).clamp(min=1e-8))
        return l_diff

    def _l_prior(self, x: torch.Tensor) -> torch.Tensor:
        """Closed-form KL  N(alpha_T * W(x), sigma_T^2 I) || N(0,I)."""
        T_idx = self.T - 1  # noqa: N806
        Wx_T = self.W(x)  # (batch, weight_dim)  # noqa: N806
        sigma_T_sq = self.sigma_sq[T_idx]  # noqa: N806
        alpha_T_sq = self.alpha_cumprod[T_idx]  # noqa: N806
        d = Wx_T.shape[-1]
        kl = 0.5 * (d * (sigma_T_sq - torch.log(sigma_T_sq) - 1.0) + alpha_T_sq * (Wx_T**2).sum(dim=-1))
        return kl

    def _l_rec(self, x: torch.Tensor) -> torch.Tensor:
        """INR-based reconstruction loss at t = 0."""
        weights = self.W(x)
        x_recon = self._inr_decode(weights)
        return 0.5 * ((x - x_recon) ** 2).sum(dim=-1)

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def sample_weight(self, n_samples: int = 1) -> torch.Tensor:
        weight_dim = self.W.weight_dim
        device = self.sqrt_alpha_cumprod.device
        theta_t = torch.randn(n_samples, weight_dim, device=device)

        for t in tqdm(range(self.T - 1, -1, -1), desc="NDM (Static) Sampling", total=self.T):
            t_idx = torch.full((n_samples,), t, dtype=torch.long, device=device)
            t_norm = torch.full((n_samples, 1), t / max(self.T - 1, 1), device=device)

            eps_hat = self.network(theta_t, t_norm)
            alpha_t = self.sqrt_alpha_cumprod[t].unsqueeze(0)
            sigma_t = self.sigma[t].unsqueeze(0)
            theta_t_hat = (theta_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

            if t == 0:
                return theta_t_hat

            s = t - 1
            s_idx = torch.full((n_samples,), s, dtype=torch.long, device=device)

            # W(x_hat) — single forward pass, no time argument needed
            x_hat_pixels = self._inr_decode(theta_t_hat)
            Wx_hat = self.W(x_hat_pixels)  # noqa: N806

            alpha_s = self.sqrt_alpha_cumprod[s].view(1, 1)
            sigma_s_sq = self.sigma_sq[s].view(1, 1)
            sigma_t_val = self.sigma[t].view(1, 1)
            alpha_t_val = self.sqrt_alpha_cumprod[t].view(1, 1)
            sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx)[0].view(1, 1)

            B = (sigma_s_sq - sigma_tilde_sq).clamp(min=0).sqrt() / sigma_t_val.clamp(min=1e-6)  # noqa: N806
            mu = alpha_s * Wx_hat + B * (theta_t - alpha_t_val * Wx_hat)
            noise = torch.randn_like(theta_t) if sigma_tilde_sq.item() > 0 else torch.zeros_like(theta_t)
            theta_t = mu + sigma_tilde_sq.clamp(min=0).sqrt() * noise

        return theta_t_hat  # safety fallback
