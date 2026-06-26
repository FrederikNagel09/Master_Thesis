from torch import nn

from src.utility.model_builders.util.latent_diffusion_builder import (
    _build_latent_diffusion,
)
from src.utility.model_builders.util.weight_diffusion_builder import (
    _build_weight_diffusion,
)


def build_model(args, data_config: dict) -> nn.Module:
    """
    Instantiate and return the model specified by args.model.

    Parameters
    ----------
    args        : argparse.Namespace with all hyperparameters.
    data_config : Dict from build_dataset() with channels/img_size/data_dim.

    Returns
    -------
    model : Untrainable nn.Module (not yet moved to device).
    """
    name = args.model.lower()

    if name == "weight_inr_diffusion":
        model = _build_weight_diffusion(args, data_config)
    elif name == "latent_inr_diffusion":
        model = _build_latent_diffusion(args, data_config)
    else:
        raise ValueError(
            f"Unknown model '{args.model}'. Choose from: 'ndm', 'inr_vae', 'ndm_inr'."
        )
    return model
