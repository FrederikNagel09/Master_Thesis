"""
run_training.py
Universal entry point for all training runs.

Usage
-----
    from src.utils.run_training import run_training
    run_training(args)

args is an argparse.Namespace.  Required fields are documented in
get_default_parser() at the bottom of this file.

Directory layout produced per run
----------------------------------
    src/train_results/{run_name}/
        config.json              - hyperparams + timing + paths
        training_graph_data.json - full loss/lr history (appended on resume)
        {run_name}.png           - training plot (overwritten every epoch)
        train_samples_ep{A}-{B}.png   - sample grid for each run segment
        weights.pt               - full checkpoint (model + optimiser + epoch)
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn

sys.path.append(".")

from src.configs.general_config import GLOBAL_DEBUG_BOOL
from src.utility.dataset_builders import build_dataset
from src.utility.general import (
    _get_device,
    _load_checkpoint,
    _load_graph_data,
    _run_dir,
    _save_checkpoint,
    _save_config,
    _save_graph_data,
)
from src.utility.model_builders import build_model
from src.utility.plotting import (
    plot_fphi_progression,
    plot_reconstruction_progression,
    plot_sample_progression,
    plot_training,
)
from src.utility.training import train

# =============================================================================
# Public API
# =============================================================================


def run_training(
    args: argparse.Namespace,
) -> nn.Module:
    """
    Full training pipeline: data → model → train → save.

    Parameters
    ----------
    args       : argparse.Namespace (see get_default_parser()).
    sample_fn  : Optional callable(model, step, device) -> None.
                 Called at 5 checkpoints during training and once at the end.

    Returns
    -------
    Trained model on CPU.
    """
    start_time = datetime.now()
    run_dir = _run_dir(args.run_name)
    device = _get_device()
    resume_path = getattr(args, "resume", None)
    use_modulation = args.use_modulation

    print("\n" + "=" * 60)
    print(f"  Run     : {args.run_name}")
    print(f"  Model   : {args.model}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Dir     : {run_dir}")
    print(f"  use_modulation: {use_modulation}")
    print("=" * 60)

    if resume_path is None:
        import glob

        # Clear stale plots and metadata from a previous run with the same name
        for fname in [
            "tqdm.log",
            "training_graph.png",
            "final_samples_ep*.png",
            "sample_progression_ep*.png",
            "fphi_progression_ep*.png",
            "reconstruction_progression_ep*.png",
            "metadata/training_graph_data.json",
            "metadata/sample_progression_*.json",
            "metadata/sample_progression_*.npy",
            "metadata/reconstruction_progression_*.json",
            "metadata/reconstruction_progression_*.npy",
            "weights/weights.pt",
        ]:
            for fpath in glob.glob(os.path.join(run_dir, fname)):
                os.remove(fpath)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print("\n[ 1 / 4 ]  Building dataset …")
    dataset, data_config = build_dataset(
        dataset_name=args.dataset,
        data_root=getattr(args, "data_root", "data/"),
        subset_frac=getattr(args, "subset_frac", 1.0),
        single_class=getattr(args, "single_class", False),
        single_class_label=getattr(args, "single_class_label", 1),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=getattr(args, "num_workers", 0),
    )
    print(f"  Batches per epoch : {len(data_loader)}")

    # Check dataset stats for sanity and for building model
    if GLOBAL_DEBUG_BOOL:
        print("==================== DEBUG: run_training.py ====================")
        print(f"  Dataset samples shape : {dataset[0][0].shape}")
        print(f"  Dataset samples range : [{dataset[0][0].min().item():.4f}, {dataset[0][0].max().item():.4f}]")
        print("================================================================")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("\n[ 2 / 4 ]  Building model …")
    model = build_model(args, data_config).to(device)

    # ── 3. Optimiser & optional resume ───────────────────────────────────────
    print("\n[ 3 / 4 ]  Setting up optimiser …")
    if False:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.0),
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.0))

    start_epoch = 0
    if resume_path is not None:
        print(f"  Resuming from checkpoint: {resume_path}")
        start_epoch = _load_checkpoint(resume_path, model, optimizer)
    else:
        print("  Training from scratch.")
    print("Training using optimizer:", type(optimizer).__name__)
    print(f"  Initial LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Determine sample filename for this run segment
    end_epoch = start_epoch + args.epochs

    progression_filename = f"sample_progression_ep{start_epoch + 1}-{end_epoch}"

    def _sample_fn(model, step, device, batch=None):
        epoch = step // len(data_loader)
        plot_sample_progression(model, args.model, epoch, run_dir, device, data_config, filename=progression_filename)
        if batch is not None:
            if args.model == "ndm":
                plot_fphi_progression(
                    model, batch, epoch, run_dir, device, data_config, filename=f"fphi_progression_ep{start_epoch + 1}-{end_epoch}"
                )
            elif args.model in ("ndm_inr", "ndm_transinr", "ndm_temporal_transinr", "ndm_static_transinr"):
                plot_reconstruction_progression(
                    model,
                    batch,
                    epoch,
                    run_dir,
                    device,
                    data_config,
                    filename=f"reconstruction_progression_ep{start_epoch + 1}-{end_epoch}",
                )

    # Load existing history for resumed runs; fresh dict otherwise
    history = _load_graph_data(run_dir)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("\n[ 4 / 4 ]  Training …\n")
    # Patch train() to save graph + graph_data after every epoch by wrapping
    # the epoch_callback hook via the existing history dict reference.
    model = train(
        model=model,
        model_type=args.model,
        data_loader=data_loader,
        epochs=args.epochs,
        device=device,
        name=args.run_name,
        lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.0),
        grad_clip=getattr(args, "grad_clip", 1.0),
        use_scheduler=getattr(args, "use_scheduler", True),
        warmup_steps=getattr(args, "warmup_steps", 5_000),
        peak_lr=getattr(args, "peak_lr", args.lr),
        log_every_n_steps=getattr(args, "log_every_n_steps", 20),
        save_dir=run_dir,
        sample_fn=_sample_fn,
        start_epoch=start_epoch,
        # Pass existing history so train() appends to it
        history=history,
        # Epoch callback: save graph data + redraw plot after every epoch
        epoch_callback=lambda h: (
            _save_graph_data(h, run_dir),
            plot_training(
                h,
                name=args.run_name,
                graph_dir=run_dir,
                use_scheduler=getattr(args, "use_scheduler", True),
            ),
        ),
        data_config=data_config,
        deactivate_progress_bar=args.deactivate_progress_bar,
    )

    print("\n  Training complete...")
    print("  Generating final sample grid …")
    #plot_final_samples(model, args.model, end_epoch, run_dir, device, data_config)
    print("Final sample grid saved to training directory.")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    end_time = datetime.now()
    metadata_path = os.path.join(run_dir, "weights")
    weights_path = _save_checkpoint(model, optimizer, end_epoch, metadata_path)
    print(f"\n  Weights saved → {weights_path}")

    _save_config(
        args=args,
        data_config=data_config,
        run_dir=run_dir,
        weights_path=weights_path,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        start_time=start_time,
        end_time=end_time,
    )

    duration = end_time - start_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n  Done.  Total time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
    print("=" * 60 + "\n")

    return model.cpu()
