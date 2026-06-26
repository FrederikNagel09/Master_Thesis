from src.models.latent_diffusion.modules.param_dit import ParamDiT
from src.models.weight_diffusion.modules.WeightnoisePredictor import (
    TransInrNoisePredictor,
)


# =============================================================================
# Printing functions:
# =============================================================================
def print_encoder_stats(model, mode="Static"):
    def count(params):
        return sum(p.numel() for p in params)

    # Count learnable parameters
    tokenizer_p = (
        count(model.tokenizer.parameters()) if hasattr(model, "tokenizer") else 0
    )
    transformer_p = (
        count(model.transformer.parameters()) if hasattr(model, "transformer") else 0
    )
    base_p = count(model.base_params.values()) if hasattr(model, "base_params") else 0
    wtoken_p = model.wtokens.numel() if hasattr(model, "wtokens") else 0
    postfc_p = (
        count(model.wtoken_postfc.parameters())
        if hasattr(model, "wtoken_postfc")
        else 0
    )

    # Handle probabilistic setup targeting the new modulation space
    logvar_p = 0
    if model.probabilistic and hasattr(model, "logvar_mlp"):
        logvar_p = count(model.logvar_mlp.parameters())

    total_learnable = (
        tokenizer_p + transformer_p + base_p + wtoken_p + postfc_p + logvar_p
    )

    # Safeguard temporal MLP scaling check
    time_embedding = 0
    if mode == "Temporal" and hasattr(model, "time_mlp"):
        time_embedding = count(model.time_mlp.parameters())
        total_learnable += time_embedding

    inr_total = model.weight_dim
    mod_total = model.modulation_dim

    print("\n" + "=" * 65)
    print(f"{'TransInrEncoder WeightEncoder Statistics':^65}")
    print("=" * 65)
    print(f"Architecture:      {model.transformer.__class__.__name__}")
    print(f"Encoder Mode:      {mode} | Probabilistic: {model.probabilistic}")
    print(f"INR Weight Dim:    {inr_total:,} (Raw Output Parameters)")
    print(f"Modulation Dim:    {mod_total:,} (Compressed Latent / Diffusion Target)")
    print(
        f"Compression Ratio: {inr_total / mod_total:.2f}x smaller space for Diffusion"
    )
    print("-" * 65)
    print("Learnable Parameters:")
    print(f"  Vision Tokenizer:     {tokenizer_p:>12,} params")
    print(f"  Main Transformer:     {transformer_p:>12,} params")
    print(f"  Weight Tokens:        {wtoken_p:>12,} params")
    print(f"  Base INR Weights:     {base_p:>12,} params")
    print(f"  Wtoken Post-FC:       {postfc_p:>12,} params")
    if model.probabilistic:
        print(f"  Logvar MLP:           {logvar_p:>12,} params")
    if mode == "Temporal" and time_embedding > 0:
        print(f"  Time MLP:             {time_embedding:>12,} params")
    print(f"  {'─'*49}")
    print(f"  Total Learnable:      {total_learnable:>12,} params")

    print("\nLayer-by-Layer Structural Breakdown:")
    print(
        f"  {'Layer Name':<15} | {'Raw INR Shape':<13} {'Total':>8} | {'Mod Shape':<10} {'Total':>7}"
    )
    print(f"  {'─'*61}")

    for name, shape in model.inr.param_shapes.items():
        l, r = model.wtoken_rng[name]  # noqa: E741
        n_groups = r - l  # noqa: F841

        # Pull raw shapes vs compact modulation tracking shapes
        inr_elements = shape[0] * shape[1]
        mod_rows, mod_g = model.modulation_shapes[name]
        mod_elements = mod_rows * mod_g

        shape_str = f"{shape[0]}x{shape[1]}"
        mod_str = f"{mod_rows}x{mod_g}"

        print(
            f"  {name:<15} | {shape_str:<13} {inr_elements:>8,} | {mod_str:<10} {mod_elements:>7,}"
        )

    print(f"  {'─'*61}")
    print(f"  {'TOTALS':<15} | {'':<13} {inr_total:>8,} | {'':<10} {mod_total:>7,}")
    print("=" * 65 + "\n")

    return total_learnable


def print_noise_predictor_stats(model, name="Noise Predictor"):  # noqa: ARG001
    def count(params):
        return sum(p.numel() for p in params)

    # Unwrap WeightTransformation wrappers to get the base model stats
    base = model

    total = count(model.parameters())

    print("\n" + "=" * 60)

    print(f"{'TransInrNoise Predictor ε_θ Statistics':^60}")
    print(f"{'(Encoder-Only DiT Architecture)':^60}")

    print("=" * 60)

    if isinstance(base, (TransInrNoisePredictor)):
        time_p = count(base.time_embed.parameters()) + count(base.time_mlp.parameters())
        token_p = count(base.token_embed.parameters())
        pos_p = base.pos_embed.numel()
        transformer_p = count(base.transformer.parameters())
        head_p = count(base.noise_head.parameters())

        print(f"  Weight Dim:   {base.weight_dim:<10} | Chunk Size: {base.chunk_size}")
        print(f"  Num Tokens:   {base.n_tokens:<10} | Padded Dim: {base.padded_dim}")
        print("-" * 60)
        print("Learnable Parameters:")
        print(f"  Time Conditioning (MLP): {time_p:>12,} params")
        print(f"  Token Embedding:         {token_p:>12,} params")
        print(f"  Positional Embedding:    {pos_p:>12,} params")
        print(f"  Transformer Blocks:      {transformer_p:>12,} params")
        print(f"  Noise Prediction Head:   {head_p:>12,} params")
        print(f"  {'─'*44}")
        print(f"  Total:                   {total:>12,} params")

    elif isinstance(base, (ParamDiT)):
        time_p = count(base.time_embed.parameters())
        tokenizer_p = count(base.tokenizer.parameters())
        detokenizer_p = count(base.detokenizer.parameters())
        blocks_p = count(base.blocks.parameters())
        norm_p = count(base.final_norm.parameters())

        n_tokens = base.tokenizer.num_params  # total weight params being tokenized
        print(f"  Hidden Dim:   {base.hidden_dim:<10} | Depth: {len(base.blocks)}")
        print(f"  Num Tokens:   {n_tokens:<10} | Time Dim: {base.time_dim}")
        print("-" * 60)
        print("Learnable Parameters:")
        print(f"  Time Embedding:          {time_p:>12,} params")
        print(f"  Tokenizer:               {tokenizer_p:>12,} params")
        print(f"  Detokenizer:             {detokenizer_p:>12,} params")
        print(f"  Transformer Blocks:      {blocks_p:>12,} params")
        print(f"  Final LayerNorm:         {norm_p:>12,} params")
        print(f"  {'─'*44}")
        print(f"  Total:                   {total:>12,} params")

    else:
        print("  Generic or Legacy Noise Predictor detected.")
        print(f"  Total parameters: {total:,}")

    print("=" * 60 + "\n")
    return total
