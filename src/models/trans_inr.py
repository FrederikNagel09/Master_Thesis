import copy
import importlib

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# ---------------------------------------------------------------------------
# Coordinate grid helper
# ---------------------------------------------------------------------------


def make_coord_grid(shape, range, device=None):
    """
    Args:
        shape : tuple of ints  e.g. (28, 28)
        range : [minv, maxv]  or [[minv_1, maxv_1], ...]
    Returns:
        grid  : (*shape, len(shape))
    """
    l_lst = []
    for i, s in enumerate(shape):
        l = (0.5 + torch.arange(s, device=device)) / s  # noqa: E741
        if isinstance(range[0], list | tuple):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        l = minv + (maxv - minv) * l  # noqa: E741
        l_lst.append(l)
    grid = torch.meshgrid(*l_lst, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    return grid


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
# TransInr — main model
# ---------------------------------------------------------------------------


class TransInr(nn.Module):
    """
    Transformer-based Implicit Neural Representation (INR) hypernetwork.

    The tokenizer now accepts **raw images** (B, C, H, W) and converts them
    into patch tokens via ImageTokenizer.  Those tokens are fed into the
    transformer encoder; a set of learnable weight-tokens (wtokens) are fed
    into the decoder.  The decoder output is used to modulate the base
    parameters of a SIREN INR, which then maps 2-D coordinates → pixel values.

    Args:
        tokenizer       : config dict for ImageTokenizer
        inr             : config dict for SIREN
        n_groups        : number of wtoken groups per INR parameter
        data_shape      : (H, W) spatial resolution of the target output
        transformer     : config dict for Transformer
        update_strategy : one of {"normalize", "scale", "identity"}
    """

    def __init__(self, tokenizer, inr, n_groups, data_shape, transformer, update_strategy="normalize", *args, **kwargs):  # noqa: ARG002
        super().__init__()

        dim = transformer["params"]["dim"]

        # ----- sub-modules -------------------------------------------------
        self.tokenizer = instantiate_from_config(tokenizer, extra_args={"dim": dim})
        self.inr = instantiate_from_config(inr)
        self.transformer = instantiate_from_config(transformer)

        # Shared coordinate grid — registered as buffer so it moves with the model
        self.register_buffer(
            "shared_coord",
            make_coord_grid(data_shape, (-1, 1)),
            persistent=False,
        )

        # ----- base INR parameters + wtoken machinery ----------------------
        self.base_params = nn.ParameterDict()
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = dict()  # noqa: C408

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

        # ----- parameter count --------------------------------------------
        nparams = (
            sum(p.numel() for p in self.transformer.parameters())
            + sum(p.numel() for p in self.tokenizer.parameters())
            + sum(p.numel() for p in self.base_params.values())
            + self.wtokens.numel()
        )
        print(f"TransInr — total hypernetwork parameters: {nparams / 1e6:.3f}M")

    # -----------------------------------------------------------------------

    def forward(self, data, coord=None, **kwargs):
        """
        Args:
            data  : (B, C, H, W)  raw image tensor
            coord : optional custom coordinate grid; uses shared_coord if None
        Returns:
            pred  : (B, C_out, H, W)  reconstructed image
        """
        # 1. Tokenise the raw image  → (B, N_patch, dim)
        dtokens = self.tokenizer(data, **kwargs)
        B = dtokens.shape[0]  # noqa: N806

        # 2. Expand wtokens to batch  → (B, N_w, dim)
        wtokens = einops.repeat(self.wtokens, "n d -> b n d", b=B)

        # 3. Transformer: image tokens → encoder memory; wtokens → decoder
        #    Supports both Transformer (enc+dec) and TransformerEncoder (enc-only)
        cls_name = self.transformer.__class__.__name__
        if cls_name == "Transformer":
            # Full encoder-decoder: image context → encoder, wtokens → decoder
            trans_out = self.transformer(src=dtokens, tgt=wtokens)
        elif cls_name == "TransformerEncoder":
            # Encoder-only fallback: concatenate and slice last n_wtoken outputs
            combined = torch.cat([dtokens, wtokens], dim=1)
            full_out = self.transformer(combined)
            trans_out = full_out[:, -self.wtokens.shape[0] :, :]
        else:
            raise ValueError(f"Unsupported transformer class: {cls_name}")

        # 4. Modulate base INR parameters with transformer output
        params = {}
        for name, shape in self.inr.param_shapes.items():  # noqa: B007
            wb = einops.repeat(self.base_params[name], "n m -> b n m", b=B)
            w = wb[:, :-1, :]  # weights
            b = wb[:, -1:, :]  # bias row

            l, r = self.wtoken_rng[name]  # noqa: E741
            x = self.wtoken_postfc[name](trans_out[:, l:r, :])
            x = x.transpose(-1, -2)  # (B, shape[0]-1, g)
            w = self.update_strategy(w, x)

            params[name] = torch.cat([w, b], dim=1)

        self.inr.set_params(params)

        # 5. Query INR at every pixel coordinate
        if coord is None:
            coord = self.shared_coord  # (H, W, 2)

        # Expand coord to batch
        if coord.dim() == 3:  # (H, W, 2)
            coord = einops.repeat(coord, "h w d -> b h w d", b=B)
        elif coord.dim() == 4:  # already batched (B, H, W, 2) — pass through
            pass

        pred = self.inr(coord)  # (B, H, W, C_out)

        # Rearrange to (B, C_out, H, W)
        if pred.dim() == 4:
            pred = pred.permute(0, 3, 1, 2).contiguous()

        return pred

    # -----------------------------------------------------------------------

    def get_last_layer(self):
        return self.inr.get_last_layer()
